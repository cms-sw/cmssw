// @(#)root/hist:$Id: RscCombinedModel.cc,v 1.1 2009/01/06 12:22:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch, Gregory.Schott@cern.ch   05/04/2008

#include <assert.h>

#include "TIterator.h"
#include "TObjArray.h"
#include "TObjString.h"

#include "PhysicsTools/RooStatsCms/interface/RscCombinedModel.h"

#include "RooFormulaVar.h"
#include "RooArgSet.h"
#include "RooGlobalFunc.h" // for RooFit::Extended()

/// For the Cint dictionaries
//ClassImp(RscCombinedModel)

/*----------------------------------------------------------------------------*/

/// Constructor
RscCombinedModel::RscCombinedModel(const char* name,
                                   const char* title,
                                   RscTotModel* model_1,
                                   RscTotModel* model_2,
                                   RscTotModel* model_3,
                                   RscTotModel* model_4,
                                   RscTotModel* model_5,
                                   RscTotModel* model_6,
                                   RscTotModel* model_7,
                                   RscTotModel* model_8,
                                   RscTotModel* model_9,
                                   RscTotModel* model_10)
    :TNamed(name,title),
     m_models_number(0),
     m_verbose(false),
     m_own_contents(false),
     m_constraints(0){

    m_pdf_buf=NULL;
    m_sigPdf_buf=NULL;
    m_bkgPdf_buf=NULL;

    m_workspace = new RooWorkspace("Combined_model_workspace");

    // The combiner istances

    TString combiner_basename=GetName();

    m_pdf_combiner=new PdfCombiner ((combiner_basename+"_combiner").Data());
    m_sigPdf_combiner=new PdfCombiner 
                                   ((combiner_basename+"_sig_combiner").Data(),
                                    m_pdf_combiner->getCategory());
    m_bkgPdf_combiner=new PdfCombiner 
                                   ((combiner_basename+"_bkg_combiner").Data(),
                                    m_pdf_combiner->getCategory());

    // add the models to the combination
    m_add(model_1);

    if (model_2!=NULL)
        m_add(model_2);
    if (model_3!=NULL)
        m_add(model_3);
    if (model_4!=NULL)
        m_add(model_4);
    if (model_5!=NULL)
        m_add(model_5);
    if (model_6!=NULL)
        m_add(model_6);
    if (model_7!=NULL)
        m_add(model_7);
    if (model_8!=NULL)
        m_add(model_8);
    if (model_9!=NULL)
        m_add(model_9);
    if (model_10!=NULL)
        m_add(model_10);
    // Add the Categories to the Workspace
//     m_workspace->import(*getCategory(kSIG));
//     m_workspace->import(*getCategory(kSIGBKG));
//     m_workspace->import(*getCategory(kBKG));


    }

/*----------------------------------------------------------------------------*/
/**
In this constructor the combined model is built according to the info typed in 
the card, so to avoid the combination in the code.
A special syntax has been implemented to fullfil these requirements. 
The combined model is specified like in the following example:
\verbatim
[my_combined_model_name]
model = combined # we tell Rsc we are going to specify a combined model
components = model_1, model_2, model_3 ... # models specified in the card, N of them
\endverbatim

**/

RscCombinedModel::RscCombinedModel(const char* combined_model_name)
    :TNamed(combined_model_name,combined_model_name),
     m_models_number(0),
     m_verbose(false),
     m_own_contents(true),
     m_constraints(0){


    m_pdf_buf=NULL;
    m_sigPdf_buf=NULL;
    m_bkgPdf_buf=NULL;

    m_workspace = new RooWorkspace("Combined_model_workspace");

    // The combiner istances

    TString combiner_basename=GetName();

    m_pdf_combiner=new PdfCombiner ((combiner_basename+"_combiner").Data());
    m_sigPdf_combiner=new PdfCombiner 
                                   ((combiner_basename+"_sig_combiner").Data(),
                                    m_pdf_combiner->getCategory());
    m_bkgPdf_combiner=new PdfCombiner 
                                   ((combiner_basename+"_bkg_combiner").Data(),
                                    m_pdf_combiner->getCategory());

    // access the card
    if (not m_is_combined(combined_model_name)){
        std::cerr << "The model does not seem to be a combined one. Aborting.\n";
        abort();
        }

    /*
    Now we parse the content of the list of models to combine. In this case the 
    RscTotModels will be owned by the RscCombined model.
    */
    TString model_components_str(m_expand_components(combined_model_name));

    std::cout << "model components: " << model_components_str.Data() << std::endl;

    // Loop on the models, allocate them and add them to the combination
    TObjArray* arr = model_components_str.Tokenize(",");
    TIterator* iter = arr->MakeIterator();
    TObjString* ostr;
    RscTotModel* model;
    while ((ostr = (TObjString*) (iter->Next()))!=NULL){
        std::cout << "Tot model: " << (ostr->GetString()).Data() << std::endl;
        model= new RscTotModel((ostr->GetString()).Data());
        m_add(model);
        }
    }

/*----------------------------------------------------------------------------*/
TString RscCombinedModel::m_expand_components(TString combined_model_name_s){
    RooStringVar model_components("components","components","");
    RooArgSet(model_components).readFromFile(RscAbsPdfBuilder::getDataCard(),
                                             0,
                                             combined_model_name_s.Data());

    TObjArray final_arr;
    // convert to TString and strip the spaces
    TString model_components_str(model_components.getVal());
    model_components_str.ReplaceAll(" ","");
    TObjArray* arr = model_components_str.Tokenize(",");
    TIterator* iter = arr->MakeIterator();
    TObjString* ostr;
    while ((ostr = (TObjString*) (iter->Next()))!=NULL){
        //std::cout << "Tot model: " << (ostr->GetString()).Data() << std::endl;
        if (m_is_combined((ostr->GetString()).Data())){
            std::cout << (ostr->GetString()).Data() 
                      << " is a combined model, inspecting further...\n";
            TString temp(m_expand_components(ostr->GetString()));
            TObjArray* temp_arr = temp.Tokenize(",");
            TIterator* temp_iter = temp_arr->MakeIterator();
            while ((ostr = (TObjString*) (temp_iter->Next()))!=NULL)
                final_arr.Add(ostr);
            }
        else
            final_arr.Add(ostr);

    }
    delete iter;

    final_arr.Expand(final_arr.GetSize());

    iter = final_arr.MakeIterator();
    TString final_str("");

    while ((ostr = (TObjString*) (iter->Next()))!=NULL)
        final_str+=ostr->GetString()+",";

    delete iter;
    final_str.Chop();

    std::cout << "Models: " << final_str << std::endl;

    return final_str;
    }

/*----------------------------------------------------------------------------*/
/**
Check if the model name is associated to a combined one.
**/
bool  RscCombinedModel::m_is_combined(const char* combined_model_name){

    // More for consistency than for the real need..
    RooStringVar model_type("model","model","");
    RooArgSet(model_type).readFromFile(RscAbsPdfBuilder::getDataCard(),
                                       0,
                                       combined_model_name);

    // if not "combined", abort!
    TString model_type_str(model_type.getVal());
    if (model_type_str!="combined")
        return false;
    else
        return true;
    }


/*----------------------------------------------------------------------------*/


/**
Add a model to the combined model. The same model can be added several times.
The pdf of the model is added to the m_pdf_combiner object as well, to be 
prepared for the pdf combination.
**/

void  RscCombinedModel::m_add(RscTotModel* model){

    if (m_verbose)
        std::cout << "[RscCombinedModel::m_add] "
                  << " adding model " << model->getName() << std::endl;

    m_models_list.Add(model);
    m_models_number++;

    // The pdf combiners
    //std::cout << "model->getPdf();\n";
    model->getPdf();
    //std::cout << "model->getVars();\n";
    model->getVars();
    //std::cout << " - adding pdf...\n";
    m_pdf_combiner->add(model->getPdf(),model->getVars());

    //std::cout << "model->getExtendedSigPdf();\n";
    model->getExtendedSigPdf();
    //std::cout << "model->getVars();\n";
    model->getVars();
    //std::cout << " - adding sigpdf...\n";
    
    m_sigPdf_combiner->add(model->getExtendedSigPdf(),model->getVars());
    //std::cout << " - adding bkgpdf...\n";
    m_bkgPdf_combiner->add(model->getExtendedBkgPdf(),model->getVars());

}

/*----------------------------------------------------------------------------*/

/**
Print the information of the RscCombinedModel instance on screen.
The "v" argument to the functions causes the information to be verbose.
**/

void RscCombinedModel::print(const char* option){

    std::cout << "\n-----------------\n"
              << "\033[7;2m " << GetName() << " Combined model\033[1;0m \n\n"
              << "The following models are present: \n";

    TIterator* iter = m_models_list.MakeIterator();
    RscTotModel* model;
    while ((model = (RscTotModel*) iter->Next())){
        std::cout << " - " << model->getName() << std::endl;
        }

    }

/*----------------------------------------------------------------------------*/

/**
Designed for debugging and didactical purposes.
**/

void RscCombinedModel::setVerbosity(bool verbose){m_verbose=verbose;}

/*----------------------------------------------------------------------------*/

/**
Get the m_pdf_combiner category. Useful for a combination with other pdfs 
external to the class instance. Maybe an interaction with PdfCombiner objects.
**/

RooCategory* RscCombinedModel::getCategory(ComponentCode code){

   if (code == kSIGBKG)
        return m_pdf_combiner->getCategory();
    if (code == kSIG)
        return m_sigPdf_combiner->getCategory();
    if (code == kBKG)
        return m_bkgPdf_combiner->getCategory();
}

/*----------------------------------------------------------------------------*/

RscCombinedModel::~RscCombinedModel(){

    // The wiseman always frees the memory

    if (m_pdf_combiner!=NULL)
        delete m_pdf_combiner;

    if (m_pdf_buf!=NULL)
        delete m_pdf_buf;


    if (m_sigPdf_combiner!=NULL)
        delete m_sigPdf_combiner;

    if (m_sigPdf_buf!=NULL)
        delete m_sigPdf_buf;


    if (m_bkgPdf_combiner!=NULL)
        delete m_bkgPdf_combiner;

    if (m_bkgPdf_buf!=NULL)
        delete m_bkgPdf_buf;

    if (m_constraints!=NULL)
        delete m_constraints;

    if (m_own_contents){
        // Let's iterate and find by name
        TIter next(&m_models_list);
        RscTotModel *model;
        while ((model = (RscTotModel*) next()))
            delete model;
        }

//     if (m_workspace!=NULL)
//         delete m_workspace;

    }

/*----------------------------------------------------------------------------*/

/**
Return the model according to the index.
**/

RscTotModel* RscCombinedModel::getModel (int index){
    return (RscTotModel*) m_models_list.At(index);
    }

/*----------------------------------------------------------------------------*/

/**
Get one model according to the index specified, seeking in the stack of models 
present inside  the object.
**/

RscTotModel* RscCombinedModel::getModel (char *name){

    TIterator* iter = m_models_list.MakeIterator();

    RscTotModel *model;
    bool found=false;

    while ((model = (RscTotModel*) iter->Next()))
        if (TString(model->getName())==TString(name)){
            found=true;
            break;
        }

    if (not found){
        std::cout << "Model " << name << " does not seem to be present.\n";
        return 0;
        }

    return model;
    }

/*----------------------------------------------------------------------------*/

/**
Find the variable in the models added to the combination by its 
name. Both the signal and the signal plus background parts are analysed.
**/

RooRealVar* RscCombinedModel::getParameter(TString name){

    RooArgSet s((this->getParameters()));

    return (RooRealVar*) s.find(name.Data());

    }

/*----------------------------------------------------------------------------*/

/**
Returns the pdf of the combination. Before returning it, the pdf is put in a 
RooWorkspace to merge the parameters who have the same name.
For the time being the RooWorkspace has to exist as long as the pdf exists, 
since it owns its components.
**/

RooAbsPdf* RscCombinedModel::getPdf(){

    if (m_pdf_buf==NULL){
        RooAbsPdf* pdf = m_pdf_combiner->getPdf();
        std::cout << "[RscCombinedModel::getPdf] Putting in workspace..\n";
//         std::cout << "********************************* Object type "
//                   << pdf->ClassName() << " obj name " << pdf->GetName()<<"\n";
//         pdf->Print("v");
//         RooArgList l(*pdf->getVariables());
//         for (int i=0;i<l.getSize();++i){
//             std::cout << "********************************* Object type "
//                       << l[i].ClassName() << " obj name " << l[i].GetName()<<"\n";
//             l[i].Print("v");
//            }
        m_workspace->import(*pdf);
        m_pdf_buf = static_cast<RooAbsPdf*>
                                        (m_workspace->function(pdf->GetName()));
        }

    return m_pdf_buf;
    }

/*----------------------------------------------------------------------------*/

/**
Returns the signal pdf of the combination. It combines the signal pdfs in the 
combination using the PdfCombiner.
**/

RooAbsPdf* RscCombinedModel::getSigPdf(){

    if (m_sigPdf_buf==NULL){
        RooAbsPdf* sigpdf = m_sigPdf_combiner->getPdf();
        m_workspace->import(*sigpdf,RooFit::RecycleConflictNodes());
        m_sigPdf_buf = static_cast<RooAbsPdf*>
                                     (m_workspace->function(sigpdf->GetName()));
        }

    return m_sigPdf_buf;

    }

/*----------------------------------------------------------------------------*/

/**
Returns the background pdf of the combination. It combines the background pdfs 
in the combination using the PdfCombiner.
**/

RooAbsPdf* RscCombinedModel::getBkgPdf(){

    if (m_bkgPdf_buf==NULL){
        RooAbsPdf* bkgpdf = m_bkgPdf_combiner->getPdf();
        m_workspace->import(*bkgpdf,RooFit::RecycleConflictNodes());
        m_bkgPdf_buf = static_cast<RooAbsPdf*>
                                     (m_workspace->function(bkgpdf->GetName()));
        }

    return m_bkgPdf_buf;

    }

/*----------------------------------------------------------------------------*/

/**
Get all the variables from the combination.
**/

RooArgList RscCombinedModel::getParameters(){

    getPdf();
    getBkgPdf();
    getSigPdf();

    return RooArgList(m_workspace->components());

//     TString par_list_name=GetName();
//     par_list_name += "_parameters";
// 
//     RooArgList pars(par_list_name.Data());
// 
//     pars.add(*getPdf()->getVariables());
// 
//     if (m_pdf_combiner!=NULL)
//          pars.add(*m_pdf_combiner->getCategory());
//     if (m_sigPdf_combiner!=NULL)
//        pars.add(*m_sigPdf_combiner->getCategory());
//     if (m_bkgPdf_combiner!=NULL)
//        pars.add(*m_bkgPdf_combiner->getCategory());
// 
//     return pars;
    }

/*----------------------------------------------------------------------------*/

/**
Get all the variables from the combination.
**/

RooArgList RscCombinedModel::getVars(){

    RooArgList vars((std::string(GetName())+"_variables").c_str());

    getPdf();getBkgPdf();getSigPdf();

    TIter next(&m_models_list);
    RscTotModel *model;

    RooRealVar* var;

    while ((model = (RscTotModel*) next())){
        RooArgList* l=model->getVars();
        for (int j=0;j<l->getSize();++j){
            var = ((RooRealVar*) &((*l)[j])); 
            if (not vars.contains(*var))
                vars.add(*var);
            }
        }
     if (m_pdf_combiner!=NULL and not vars.contains(*m_pdf_combiner->getCategory()))
         vars.add(*m_pdf_combiner->getCategory());
     if (m_sigPdf_combiner!=NULL and not vars.contains(*m_sigPdf_combiner->getCategory()))
        vars.add(*m_sigPdf_combiner->getCategory());
     if (m_bkgPdf_combiner!=NULL and not vars.contains(*m_bkgPdf_combiner->getCategory()))
       vars.add(*m_bkgPdf_combiner->getCategory());

    return vars;
    }

/*----------------------------------------------------------------------------*/

/**
Finds variable in RooArgSet according to its name. Simply get the iterator on 
the content, go through it and check if the name of the variable corresponds to 
the requested one.
**/

RooRealVar* RscCombinedModel::m_find_in_set(RooArgSet* set,TString name){

    TIterator* iter = set->createIterator();
    RooRealVar* param = (RooRealVar*) iter->Next();
    while (param!=0) {
        if (name==param->GetName())
            break;
        //continue the iteration
        param = (RooRealVar*) iter->Next();
        }
    delete iter;
    return param;
    }

/*----------------------------------------------------------------------------*/

RooArgList RscCombinedModel::getConstraints(){

    if (m_pdf_buf==NULL)
        getPdf();

    if (m_constraints!=NULL)
        return *m_constraints;

    m_constraints=new RooArgList("Constraints");

//     for (int j=0;j<getSize();++j){
//         RooArgList l(*(getModel(j)->getConstraints()));
//         for (int i=0;i<l.getSize();++i)
//             if (not m_constraints->contains(l[i]))
//                 m_constraints->add(l[i]);
//         }

    TString constraint_class_name="Constraint";
    TIterator* iter = m_workspace->componentIterator();
    RooAbsArg* branch ;
    while ((branch=(RooAbsArg*)iter->Next())){
        TString class_name(branch->ClassName());
        if (class_name==constraint_class_name)
            m_constraints->add(*(static_cast<Constraint*>(branch)));
        }

    return *m_constraints;
}

/*----------------------------------------------------------------------------*/

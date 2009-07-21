#include "RooProdPdf.h"
#include "RooFormulaVar.h"
#include "RooExtendPdf.h"

#include "PhysicsTools/RooStatsCms/interface/RscMultiModel.h"
#include "PhysicsTools/RooStatsCms/interface/RscBaseModel.h"


/*----------------------------------------------------------------------------*/

RscMultiModel::RscMultiModel(TString theName, RooArgList& theVars)
{
  if (verbose())
  cout << "Instanciating a " << ClassName() << " object named " << theName << "\n";
  _name = theName;
  _vars = &theVars;
  pdfs = 0; ownPDFs = kFALSE;
  //_yield = new RooRealVar(_name+"_yield","yield of "+_name,0);
  readDataCard();
}

/*----------------------------------------------------------------------------*/

RscMultiModel::~RscMultiModel() {
  // destructor
  if (verbose())
  cout << "Destructing the " << _name << " object\n";
  if ( ownPDFs) 
    delete pdfs;
  if (_non_ext_pdf!=NULL)
    delete _non_ext_pdf;

}

/*----------------------------------------------------------------------------*/

void RscMultiModel::buildPdf()
{

  // list of PDFs for each of the discriminating variable
  ownPDFs = kTRUE;
  pdfs = new TList();
  pdfs->SetOwner();
  TIterator* iter = _vars->createIterator();
  RooRealVar* variable = (RooRealVar*) iter->Next();
  while ( variable!=0 ) {
    TString variable_name = variable->GetName();
    if (verbose())
        cout << "Discriminating variable: " << variable_name << endl;
    RscBaseModel* bld = new RscBaseModel(_name+"_"+variable_name,*variable);
    pdfs->Add(bld);
    bld->getPdf();
    variable = (RooRealVar*) iter->Next();
  }
  delete iter;
  delete variable;


  RooArgList pdfList;
  TIter next(pdfs);
  while (TObject *obj = next()) {
    //RscBaseModel* bld = dynamic_cast<RscBaseModel*>(obj);
    RscBaseModel* bld = static_cast<RscBaseModel*>(obj);
    assert(bld);
    pdfList.add(*bld->getPdf());
  }
  if (verbose()){
    cout << "list of PDFs:\n";
    pdfList.Print("v");
    }
  // Extend the pdf directly at this level and not the the RscTotModel Level
  _non_ext_pdf = new RooProdPdf(_name+"_multi_non_ext","",pdfList);

  if (verbose())
    std::cout << "DEBUG\n"
                << " - RscMultiModel Yield:";
  //_yield->Print("v");

  _thePdf = new RooExtendPdf ((_name+"_multi").Data(),"",*_non_ext_pdf,*_yield);
}

/*----------------------------------------------------------------------------*/

RooFormulaVar* RscMultiModel::getYield() {
  return _yield;
}

/*----------------------------------------------------------------------------*/

void RscMultiModel::readDataCard()
{
  // read the data card

    /* YIELD SECTION */

    /* 
    Seek in the section if the "yield_factors_number" is present.
    The code -1000 is the one used to adopt a single component sig yield.
    */
    RooRealVar yield_factors_number("yield_factors_number","",-1000);
    RooArgSet(yield_factors_number).readFromFile(getDataCard(), 0, _name);

    // if the user typed an improper number, he fails!
    //assert(yield_factors_number.getVal() == -1000);

    /*
    If the number if -1000, let's seek for a single yield.
    A conservative choice to keep the same syntax like:

    [hzz_4e_sig]
    hzz_4e_sig_yield = 38.20 L(0 - 200)

     */
    if (yield_factors_number.getVal() == -1000){

        // Read the yield as a RooRealVar

        RooRealVar* single_yield;

        readParameter(_name+"_yield",
                      "single yield component of "+_name,
                      _name,
                      getDataCard(),
                      single_yield);



        // Put it into the RooFormulaVar of the yield 
        TString single_yield_new_name=single_yield->GetName();
        if (verbose())
            cout << "A formulavar called "<< (_name+"_yield_formula").Data()<<endl;
        _yield = new RooFormulaVar ((_name+"_yield_formula").Data(),
                                    ("yield of "+_name).Data(),
                                    single_yield_new_name.Data(),
                                    RooArgList(*single_yield));
        }

    /*
    Here we have more components of the Yield. No problem. We read them and 
    then put them together in the formulavar multiplying them. So we read 
    something like this:

    [hzz_4e_sig]

    yield_factors_number = 3

    yield_factor_1 = hzz_4e_sig_lumi
    hzz_4e_sig_lumi = 1 C

    yield_factor_2 = hzz_4e_sig_xsec
    hzz_4e_sig_xsec = 3820 L(0 - 20000)

    yield_factor_3 = hzz_4e_sig_eff
    hzz_4e_sig_eff = 0.01 C

    */
    else{


        RooArgList factors_list("factor_list");
        TString formula="";
        if (verbose())
            cout << "A formulavar called "<< (_name+"_yield_formula").Data()<<endl;
        // Read the yield factor names, then the actual variables
        TString factor_name="";
        TString factor_name_base="yield_factor_";
        for (int factor_index=1;
             factor_index<=yield_factors_number.getVal();
             ++factor_index){

            // the name
            factor_name=factor_name_base;
            factor_name+=factor_index;
            RooStringVar name_var(factor_name,"","");
            RooArgSet(name_var).readFromFile(getDataCard(), 0, _name);
            if (verbose())
                cout << name_var.getVal() << " Read ...\n";

            // add it to the formula
            formula+=name_var.getVal();
            formula+=" * ";

            // the variable
            RooRealVar* factor;//=new RooRealVar (name_var.getVal(),"",0);
            //RooArgSet(*factor).readFromFile(getDataCard(), 0, _name);

            readParameter(name_var.getVal(),
                          name_var.getVal(),
                          _name,
                          getDataCard(),
                          factor);


            // add it to the list
            factors_list.add (*factor);

//             if (_caller_vars!=NULL)
//                 _caller_vars->add(*factor);
            }

        // remove the bloody " *" at the end of the formula and print it!
        formula.Chop();
        formula.Chop();
        if (verbose())
            cout << "Yield formula is: " << formula.Data() << endl;

        // Build the yield
        _yield = new RooFormulaVar ((_name+"_yield_formula").Data(),
                                    ("yield of "+_name).Data(),
                                    formula.Data(),
                                    factors_list);
        }


//   if (getDataCard()) {
//     //RooArgSet(*_yield).readFromFile(getDataCard(), 0, _name);
//     readParameter(_yield->GetName(),
//                   _yield->GetTitle(),
//                   _name,
//                   getDataCard(),
//                   _yield);
        
    }

/*----------------------------------------------------------------------------*/

void RscMultiModel::writeDataCard(ostream& out)
{
  // write the data card
  out << "[" << _name << "]" << endl;
  RooArgSet(*_yield).writeToStream(out,false);
}

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

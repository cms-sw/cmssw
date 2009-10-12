// @(#)root/hist:$Id: PdfCombiner.cc,v 1.3 2009/04/15 11:10:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch, Gregory.Schott@cern.ch   05/04/2008

#include "PhysicsTools/RooStatsCms/interface/PdfCombiner.h"
#include "TIterator.h"

/// For the Cint dictionaries

/*----------------------------------------------------------------------------*/

/// Default Constructor

/**
Default names are assigned to the member objects.
**/

PdfCombiner::PdfCombiner():
    TNamed("My category","My category"),
    m_external_category(false),
    m_verbose(false),
    m_combined_model_buffer(NULL){

    m_model_collection = new RooArgList("My combination models");
    m_variables_collection = new RooArgList("My combination variables");

    m_category = new RooCategory ("My category",
                                  "discriminator between the modes");

    }

/*----------------------------------------------------------------------------*/

/// The constructor

PdfCombiner::PdfCombiner(std::string name):
    TNamed(name.c_str(),name.c_str()),
    m_external_category(false),
    m_verbose(false),
    m_combined_model_buffer(NULL){

    m_model_collection = new RooArgList((name+" models").c_str());
    m_variables_collection = new RooArgList((name+" variables").c_str());

    m_category = new RooCategory (name.c_str(),
                                  "discriminator between the modes");
    }

/*----------------------------------------------------------------------------*/

/// The constructor

/**
Designed to import an external RooCategory into the combination.
**/

PdfCombiner::PdfCombiner(std::string name, RooCategory* cat):
    TNamed(name.c_str(),name.c_str()),
    m_external_category(true),
    m_verbose(false),
    m_combined_model_buffer(NULL){

    m_model_collection = new RooArgList((name+" models").c_str());
    m_variables_collection = new RooArgList((name+" variables").c_str());

    m_category=cat;

    }

/*----------------------------------------------------------------------------*/

/**
Add a distribution to the combined model, together with its variable.
The same distribution can be added several times. The variables are added only 
one time.
**/

int PdfCombiner::add(RooAbsPdf* distrib, RooArgList* vars_list){

   // Add to the arrays the models
    if (m_verbose)
        std::cout << "[PdfCombiner::add] Adding pdf to the collection...\n";
    m_model_collection->add(*distrib);
    if (m_verbose)
        std::cout << "[PdfCombiner::add] Pdf added to the collection...\n";
    int return_code=1;

    // Check if the variable is present. If not add it.
    for(int i=0;i<vars_list->getSize();++i)
        if (not m_variables_collection->contains((*vars_list)[i])){
            m_variables_collection->add((*vars_list)[i]);
            return_code=2;
            }
    /*
    Add a category (a naming rule is adopted for the categories).
    If the category was given in the constructor, do not define a new type.
    */

    if (not m_external_category){
        char cat_name[5];
        sprintf(cat_name, "%d",m_model_collection->getSize()-1);
        if (m_verbose)
            std::cout << "[PdfCombiner::add]  Adding type: "
                      << cat_name
                      << std::endl;
        m_category->defineType(cat_name,m_model_collection->getSize()-1);
        }

    else {
        int category_size=0;
        TIterator* it = m_category->typeIterator();
        TObject* obj=it->Next();
        while (obj!=NULL){
            ++category_size;
            obj=it->Next();
            }
        if (category_size==m_model_collection->getSize())
            std::cout << "[PdfCombiner::add] WARNING: "
                      << "adding more pdfs than types in the category!";
        }

    if (m_verbose){
        std::cout << "[PdfCombiner::add] Model "
                  << distrib->GetName();
        if (return_code == 2){
            std::cout << " and variable ";
            for(int i=0;i<vars_list->getSize();++i)
                std::cout << (*vars_list)[i].GetName() << " ";
            }
        std:: cout << " added successfully!\n";
        }

    // Free the memory of the combined buffer if not already freed


    if (m_combined_model_buffer!=0){
        if (m_verbose)
            std::cout <<  "[PdfCombiner::add] Freeing the buffer\n";
        delete m_combined_model_buffer;
        m_combined_model_buffer=0;
        }

    return return_code;
    }

/*----------------------------------------------------------------------------*/

/**
Add a distribution to the combined model, together with its variable.
The same distribution can be added several times. The variables are added only 
one time.
**/

int PdfCombiner::add(RooAbsPdf* distrib, RooRealVar* x){
    RooArgList l(*x);
    return add(distrib,&l); 
}

/*----------------------------------------------------------------------------*/

/**
The "v" argument to the functions causes the information to be verbose.
**/

void PdfCombiner::Print(const char* option){

    std::cout << "\n-----------------\n"
              << "\033[7;2m " << GetName() << " Combined model\033[1;0m \n\n";

    std::cout << "  Category:";
    if (m_external_category)
        std::cout << "  --> The category has been imported from somewhere.";
    m_category->Print(option);

    std::cout << "\n  Models:\n";
    m_model_collection->Print(option);

    std::cout << "\n  Variables:\n";
    m_variables_collection->Print(option);

    }

/*----------------------------------------------------------------------------*/

/// Return the combined model distribution.

/**
Codes:
 - -1 for the total, combined pdf 
 - 0,1,2,... for a single component
WARNING: it is up to the user to deallocate the memory of the returned pdf.
An internal mechanism deallocates the memory dedicated to the total pdf in case 
a new channel is added to the combination.
**/

RooAbsPdf* PdfCombiner::getPdf(int index){

    // Case 1: a totally messed up index < than -1
    if (index<-1 ){
        std::cout << "[PdfCombiner::GetPdf] Index for pdf is " 
                  << index << std::endl;
        abort();
        }

    // Case 2: a positive index > the number of stored pdfs
    if (index>=m_model_collection->getSize()){
        std::cout << "[PdfCombiner::GetPdf] Index for pdf is "
                  << index
                  << " while the total number of models pdfs is "
                  << m_model_collection->getSize()
                  << std::endl;
        abort();
        }

    // Case 3: a good index for one component
    if (index!=-1)
        return (RooAbsPdf*) &(*m_model_collection)[index];


    // Case 4: the combined pdf

    //either the pdf is already there, either you build it
    if (m_combined_model_buffer==0){
        if (m_verbose)
            std::cout << "[PdfCombiner::GetPdf] " 
                      << " A new addidion occourred: recreating the Pdf ...\n";

        // Allocate the pdf for the simultaneous fit
        TString sim_pdf_name=GetName();
        sim_pdf_name+="_simoultaneus_pdf";
        m_combined_model_buffer= 
            new RooSimultaneous(sim_pdf_name.Data(),
                                "PDF for simultaneous fit",
                                *m_category);
        /* 
        Add to it all the models present. 
        There is a naming collection for the categories: they are called with an 
        integer index. "0" for the first, "1" for the second ..
        */

        char dummy[10];
        for (int i=0;i<m_model_collection->getSize();++i){
            sprintf(dummy,"%d",i);
            RooAbsPdf* thepdf = (RooAbsPdf*) &(*m_model_collection)[i];
            m_combined_model_buffer->addPdf( *thepdf ,dummy);
            }
        }

    return m_combined_model_buffer;

}

/*----------------------------------------------------------------------------*/

/**
The list of the variables of the combined channels together with the category 
are put in a RooARgList and they are returned. 
**/

RooArgList PdfCombiner::getVars(){
    RooArgList vars(*m_variables_collection);
    vars.add(*m_category);
    return vars;
}

/*----------------------------------------------------------------------------*/

/**
Designed for debugging and didactical purposes.
**/

void PdfCombiner::setVerbosity(bool verbose){
    m_verbose=verbose;
}

/*----------------------------------------------------------------------------*/

RooCategory* PdfCombiner::getCategory(){

    return m_category;
}

/*----------------------------------------------------------------------------*/

PdfCombiner::~PdfCombiner(){

    delete m_variables_collection;
    delete m_model_collection;
    if (not m_external_category)
        delete m_category;
    }

/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

// @(#)root/hist:$Id: LikelihoodCalculator.cc,v 1.1 2009/01/06 12:22:43 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#include "assert.h"
#include <iostream>

#include "PhysicsTools/RooStatsCms/interface/LikelihoodCalculator.h"

#include "RooMinuit.h"
#include "RooFitResult.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooGlobalFunc.h" // RooFit::XXXX

#include "TIterator.h"


/// To build the cint dictionaries
//ClassImp(LikelihoodCalculator)

/*----------------------------------------------------------------------------*/

/**
Plain constructor, just model and dataset.
**/

LikelihoodCalculator::LikelihoodCalculator (RooAbsPdf& model_pdf, 
                                            RooAbsData& dataset):
    m_verbose(false){

    m_base_nll = new RooNLLVar ("base_nll", 
                                "The base of the negative log likelihood",
                                model_pdf,
                                dataset,
                                RooFit::Extended(),
                                RooFit::Verbose());// We wanna it extended

    // No constraint :)
    m_nll_constr = new RooFormulaVar ("nll_constr",
                                      "@0",
                                      RooArgList(*m_base_nll));

    }

/*----------------------------------------------------------------------------*/

/** 
Constructor that takes into account the penalty terms.
**/

LikelihoodCalculator::LikelihoodCalculator (RooAbsPdf& model_pdf, 
                                            RooAbsData& dataset,
                                            TString& penalty_formula,
                                            RooArgList& penalty_terms):
    m_verbose(false){

    m_base_nll = new RooNLLVar ("base_nll", 
                                "The base of the negative log likelihood",
                                model_pdf,
                                dataset,
                                RooFit::Extended(),
                                RooFit::Verbose());


    RooArgList constr_nll_terms(*m_base_nll);
    constr_nll_terms.add(penalty_terms);

    m_nll_constr = new RooFormulaVar ("nll_constr",
                                      ("@0"+penalty_formula).Data(),
                                      constr_nll_terms);
    }


/*----------------------------------------------------------------------------*/

void LikelihoodCalculator::m_save_params_values (RooFormulaVar* nll){

    // Get the vars frmo the nll and save their value
    RooArgSet* vars=nll->getVariables();
    if (m_verbose)
        std::cout << "[LikelihoodCalculator::m_save_params_values] "
                  << vars->getSize() << " variables detected...\n";

    TIterator* var_iter=vars->createIterator();
    RooRealVar* var=(RooRealVar*)var_iter->Next();

    while(var!=0){
         m_original_var_values.push_back(var->getVal());
         var=(RooRealVar*)var_iter->Next();
        }

    delete var_iter;
    delete vars;

    }

/*----------------------------------------------------------------------------*/

void LikelihoodCalculator::m_restore_params_values (RooFormulaVar* nll){

    int var_index=0;

    // Set vars of the model
    RooArgSet* vars=nll->getVariables();
    TIterator* var_iter=vars->createIterator();
    RooRealVar* var=(RooRealVar*)var_iter->Next();

    while(var!=0){
        if (m_verbose)
            std::cout << "[LikelihoodCalculator::m_restore_params_values] "
                      << " - Parameter "
                      << var->GetName()
                      << ": " << var->getVal() ;

        var->setVal(m_original_var_values[var_index]);

        if (m_verbose)
            std::cout << " --> " << var->getVal() << std::endl;

        ++var_index;
        var=(RooRealVar*)var_iter->Next();

        }
    delete var_iter;
    delete vars;
    }

/*----------------------------------------------------------------------------*/

LikelihoodCalculator::~LikelihoodCalculator (){

    if (m_base_nll!=NULL)
        delete m_base_nll;

    if (m_nll_constr!=NULL)
        delete m_nll_constr;

m_original_var_values.clear();

    }

/*----------------------------------------------------------------------------*/

/**
Get the value of the likelihood, if the minimise flag is true it returns the 
minimum NLL.
**/

double LikelihoodCalculator::getValue(bool minimise){

    // Save the values of the variables
    if (minimise)
        m_save_params_values(m_nll_constr);

    // An istance of minuit
    RooMinuit* minuit = new RooMinuit(*m_nll_constr);

    if (minimise)
        minuit->migrad();

    RooFitResult* result=0; 
    if (minimise)
        result = minuit->save();

    if (m_verbose and minimise)
        result->Print("v");

    double value=m_nll_constr->getVal();

    if (minimise){
        m_restore_params_values(m_nll_constr);
        delete result;
        }

    delete minuit;

    return value;

}

/*----------------------------------------------------------------------------*/

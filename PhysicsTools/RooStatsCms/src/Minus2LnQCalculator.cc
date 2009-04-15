// @(#)root/hist:$Id: Minus2LnQCalculator.cc,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#include "assert.h"
#include <iostream>

#include "PhysicsTools/RooStatsCms/interface/Minus2LnQCalculator.h"


/*----------------------------------------------------------------------------*/

Minus2LnQCalculator::Minus2LnQCalculator (RooAbsPdf& sb_model_pdf,
                                          RooAbsPdf& b_model_pdf,
                                          RooAbsData& dataset):
    m_verbose(false){
    // The likelihood calculators
    m_Lcalcs[0] = new LikelihoodCalculator(sb_model_pdf,
                                           dataset);

    m_Lcalcs[1] = new LikelihoodCalculator(b_model_pdf,
                                           dataset);




    }

/*----------------------------------------------------------------------------*/

Minus2LnQCalculator::Minus2LnQCalculator (RooAbsPdf& sb_model_pdf,
                                          RooAbsPdf& b_model_pdf,
                                          TString& formula,
                                          RooArgList terms,
                                          RooAbsData& dataset):
    m_verbose(false){

    m_init_with_penalties(sb_model_pdf,
                          b_model_pdf,
                          formula,
                          terms,
                          formula,
                          terms,
                          dataset);
    }

/*----------------------------------------------------------------------------*/

Minus2LnQCalculator::Minus2LnQCalculator (RooAbsPdf& sb_model_pdf,
                                          RooAbsPdf& b_model_pdf,
                                          TString& sb_formula,
                                          RooArgList sb_terms,
                                          TString& b_formula,
                                          RooArgList b_terms,
                                          RooAbsData& dataset):
    m_verbose(false){

    m_init_with_penalties(sb_model_pdf,
                          b_model_pdf,
                          sb_formula,
                          sb_terms,
                          b_formula,
                          b_terms,
                          dataset);
    }

/*----------------------------------------------------------------------------*/

void Minus2LnQCalculator::m_init_with_penalties(RooAbsPdf& sb_model_pdf,
                                                RooAbsPdf& b_model_pdf,
                                                TString& sb_formula,
                                                RooArgList sb_terms,
                                                TString& b_formula,
                                                RooArgList b_terms,
                                                RooAbsData& dataset){

    m_Lcalcs[0] = new LikelihoodCalculator(sb_model_pdf,
                                           dataset,
                                           sb_formula,
                                           sb_terms);

    m_Lcalcs[1] = new LikelihoodCalculator(b_model_pdf,
                                           dataset,
                                           b_formula,
                                           b_terms);

    }

/*----------------------------------------------------------------------------*/
/**
Part of the destructor.
**/
void Minus2LnQCalculator::free(int j){

    for (int i=0;i<j;++i)
            delete m_Lcalcs[i];
    }


/*----------------------------------------------------------------------------*/


Minus2LnQCalculator::~Minus2LnQCalculator (){
    free(2);
    }

/*----------------------------------------------------------------------------*/

/**
Get the value of minus two times logarithm of the ratio of the likelihoods.
**/

double Minus2LnQCalculator::getValue(bool minimise){

    double sb_nll= m_Lcalcs[0]->getValue(minimise);
    double b_nll = m_Lcalcs[1]->getValue(minimise);
    double minus2lnQ = 2*(sb_nll-b_nll); // log function properties

    if (m_verbose)
        std::cout << "[Minus2LnQCalculator::getValue] Summary:\n"
                  << " * SB negative log likelihood: " << sb_nll << std::endl
                  << " *  B negative log likelihood: " << b_nll << std::endl
                  << "   --> -2lnQ = " << minus2lnQ << std::endl;

    return minus2lnQ;

}

/*----------------------------------------------------------------------------*/

/**
Get the value of sqrt(2lnQ).
**/

double Minus2LnQCalculator::getSqrtValue(bool minimise){

    double minus2lnQ=-1*getValue(minimise);
    double sqrt_minus2lnQ;
    if (minus2lnQ>0)
        sqrt_minus2lnQ = sqrt(minus2lnQ);
    else{
        if (m_verbose)
            std::cout << "[Minus2LnQCalculator::getSqrtValue] -2lnQ = "
                      << minus2lnQ
                      << " putting the square root to 0...\n";
        double sqrt_minus2lnQ=0.;
        }

    if (m_verbose)
        std::cout << "[Minus2LnQCalculator::getSqrtValue] Summary:\n"
                  << "   --> sqrt(2lnQ) = " << sqrt_minus2lnQ << std::endl;

    return sqrt_minus2lnQ;

}

/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

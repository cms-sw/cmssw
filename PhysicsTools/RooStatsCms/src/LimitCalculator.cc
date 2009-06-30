// @(#)root/hist:$Id: LimitCalculator.cc,v 1.3 2009/04/15 11:10:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include <iostream>

#include "PhysicsTools/RooStatsCms/interface/LimitCalculator.h"
#include "PhysicsTools/RooStatsCms/interface/Minus2LnQCalculator.h"

#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooGlobalFunc.h"

#include "RooRandom.h"

#include "PhysicsTools/RooStatsCms/interface/Rsc.h" // the Rsc Namespace


/*----------------------------------------------------------------------------*/

LimitCalculator::LimitCalculator(const char* name,
                                 const char* title,
                                 RooAbsPdf* sb_model,
                                 RooAbsPdf* b_model,
                                 RooArgList* variables,
                                 ConstrBlockArray* c_array)
    :StatisticalMethod(name,title,true),
    m_sb_model(sb_model),
    m_b_model(b_model),
    m_variables(variables),
    m_c_array(c_array){
    if (c_array!=NULL){
        // Get the penalty terms of the likelihood
        m_NLL_string=c_array->getNLLstring();
        m_NLL_terms.add(c_array->getNLLterms());

        m_Bkg_NLL_string=c_array->getBkgNLLstring();
        m_Bkg_NLL_terms.add(c_array->getBkgNLLterms());
        }

    //m_NLL_terms.Print("v");

    }

/*----------------------------------------------------------------------------*/


LimitCalculator::~LimitCalculator(){
    }


/*----------------------------------------------------------------------------*/

LimitResults* LimitCalculator::calculate(TH1* data, 
                                         unsigned int n_toys,
                                         bool fluctuate){

    if (is_verbose())
        std::cout << "[LimitCalculator::Calculate] "
                  << " Converting Histogram to RooDataHist ...\n";

    RooDataHist data_hist("Histo as datahist",
                          "Histo as datahist",
                          *m_variables,
                          data);

    return calculate(&data_hist,n_toys,fluctuate);
    }


/*----------------------------------------------------------------------------*/

LimitResults* LimitCalculator::calculate(RooTreeData* data,
                                         unsigned int n_toys,
                                         bool fluctuate){

    // Calculate the value of -2lnQ on the data
    Minus2LnQCalculator calc (*m_sb_model,
                              *m_b_model,
//                               m_NLL_string,
//                               m_NLL_terms,
//                               m_Bkg_NLL_string,
//                               m_Bkg_NLL_terms,
                              *data);

    float m2lnq_data=calc.getValue(false);

    // Allocate the containers for the results
    std::vector<float> b_vals;
    b_vals.reserve(n_toys);

    std::vector<float> sb_vals;
    sb_vals.reserve(n_toys);

    m_do_toys(b_vals,sb_vals,n_toys,fluctuate);

    LimitResults* res = new LimitResults(GetName(),
                                         GetTitle(),
                                         sb_vals,
                                         b_vals,
                                         m2lnq_data);

    return res;
    }
/*----------------------------------------------------------------------------*/

LimitResults* LimitCalculator::calculate(unsigned int n_toys,
                                         bool fluctuate){

    // Allocate the containers for the results
    std::vector<float> b_vals;
    b_vals.reserve(n_toys);

    std::vector<float> sb_vals;
    sb_vals.reserve(n_toys);

    m_do_toys(b_vals,sb_vals,n_toys,fluctuate);

    LimitResults* res = new LimitResults(GetName(),
                                         GetTitle(),
                                         sb_vals,
                                         b_vals,
                                         0);

    return res;
    }

/*----------------------------------------------------------------------------*/
/**
FIXME: explain here:
 - fluctuations
 - Yields
 - poisson
**/
void LimitCalculator::m_do_toys(std::vector<float>& b_vals,
                                std::vector<float>& sb_vals,
                                unsigned int n_toys,
                                bool fluctuate){

    assert(n_toys > 0);

    RooTreeData* b_data;
    RooTreeData* sb_data;

    bool check_sb=false;
    bool check_b=false;

//     unsigned int b_data_size=0;
//     unsigned int sb_data_size=0;

    for (unsigned int i=0;i<n_toys;++i){

        // instrument teh code to check the rndmseed
        if (n_toys < 100 and (i)%1==0)
        std::cout << "Toy Num " << i << " - " << gRandom->GetSeed()<< std::endl;

        if (is_verbose() and (i+1)%500==0)
            std::cout << "\033[1;31m" 
                      << "====================================\n"
                      << "   Toy number num. " << i+1 << std::endl 
                      << "==================================== "
                      << "\033[1;0m \n";

        if (fluctuate)
            m_c_array->fluctuate();




         // Generate the data sets
         // bug in roofit for generation





        b_data = static_cast<RooTreeData*> (m_b_model->generate(*m_variables,RooFit::Extended()));

        if (b_data==NULL){
              std::cerr << "\n\n\n\nEmpty B dataset!\n\n\n\n\n";
              RooDataSet* b_data_dummy=new RooDataSet("an empty one","",*m_variables);
              b_data = static_cast<RooTreeData*>(new RooDataHist ("datab","",*m_variables,*b_data_dummy));
              delete b_data_dummy;
              check_b=true;
             }


//        std::cout << "B model\n";
//        std:: cout << (m_b_model->getVariables()->find("bkg_yield")) << std::endl;

//         m_c_array->print();

        if (fluctuate)
            m_c_array->fluctuate();
//         b_data_size=RooRandom::randomGenerator()->Poisson(m_b_model->expectedEvents(*m_variables));
//         if (b_data_size!=0)
//             b_data=static_cast<RooTreeData*> (m_b_model->generate(*m_variables,b_data_size, b_data_size));
//         else{
//             std::cerr << "\n\n\n\nEmpty B dataset!\n\n\n\n\n";
//             RooDataSet* b_data_dummy=new RooDataSet("an empty one","",*m_variables);
//             b_data = static_cast<RooTreeData*>(new RooDataHist ("datab","",*m_variables,*b_data_dummy));
//             check_b=true;
//             delete b_data_dummy;
//             }

         sb_data = static_cast<RooTreeData*> (m_sb_model->generate(*m_variables,RooFit::Extended()));

         if (sb_data==NULL){
              std::cerr << "Empty SB dataset!\n";
              RooDataSet* sb_data_dummy=new RooDataSet("an empty one","",*m_variables);
              sb_data = static_cast<RooTreeData*>(new RooDataHist ("datasb","",*m_variables,*sb_data_dummy));
              delete sb_data_dummy;
              check_sb=true;
             }

//         sb_data_size=RooRandom::randomGenerator()->Poisson(m_sb_model->expectedEvents(*m_variables));
//         if (sb_data_size!=0)
//             sb_data=static_cast<RooTreeData*> (m_sb_model->generate(*m_variables,sb_data_size));
//         else{
//             std::cerr << "\n\n\n\nEmpty SB dataset!\n\n\n\n\n";
//             RooDataSet* sb_data_dummy=new RooDataSet("an empty one","",*m_variables);
//             sb_data = static_cast<RooTreeData*>(new RooDataHist ("datasb","",*m_variables,*sb_data_dummy));
//             check_sb=true;
//             delete sb_data_dummy;
//             }

        //m_sb_model->Print("v");
//         std::cout << "SB model\n";
//         std::cout << ((RooRealVar*)m_b_model->getVariables()->find("bkg_yield"))->getVal() << std::endl;

         if (fluctuate)
             m_c_array->restore();






        // Calculate by hand the 4 likelihoods
        RooNLLVar sb_sb_nll ("sb_sb_nll","sb_sb_nll",*m_sb_model,*sb_data,RooFit::Extended());
        RooNLLVar b_sb_nll ("b_sb_nll","b_sb_nll",*m_b_model,*sb_data,RooFit::Extended());

        double m2lnQ=2*(sb_sb_nll.getVal()-b_sb_nll.getVal());
        sb_vals.push_back(m2lnQ);

        if (check_sb)
            std::cerr << "Dataset was empty. lnQ = " << b_sb_nll.getVal() << " - " << sb_sb_nll.getVal() << std::endl;

        RooNLLVar sb_b_nll ("sb_b_nll","sb_b_nll",*m_sb_model,*b_data,RooFit::Extended());
        RooNLLVar b_b_nll ("b_b_nll","b_b_nll",*m_b_model,*b_data,RooFit::Extended());

        m2lnQ=2*(sb_b_nll.getVal()-b_b_nll.getVal());
        b_vals.push_back(m2lnQ);

        if (check_b){
            double bnll=b_b_nll.getVal();
            double sbnll=sb_b_nll.getVal();
            std::cerr << "Dataset was empty."
                      << "-lnQ = " << sbnll
                      << " - " << bnll
                      << "\n-2lnQ = " << m2lnQ
                      << std::endl;
            }

        delete sb_data;
        delete b_data;

        check_sb=false;
        check_b=false;

//          // SB
//          Minus2LnQCalculator calc_sb (*m_sb_model,
//                                       *m_b_model,
//                                       *sb_data);
//  
//          sb_vals.push_back(calc_sb.getValue(false));
// 
//          // B
//          Minus2LnQCalculator calc_b (*m_sb_model,
//                                      *m_b_model,
//                                      *b_data);
// 
//          b_vals.push_back(calc_b.getValue(false));


        }// end loop on the toys
    }

/*----------------------------------------------------------------------------*/

void LimitCalculator::print(const char* options){

    std::cout << "\nSignal plus background model:\n";
    m_sb_model->Print(options);

    std::cout << "\nBackground model:\n";
    m_b_model->Print(options);

    std::cout << "\nVariables\n";
    m_variables->Print(options);

    std::cout << "\nArray of constraints\n";
    m_c_array->print(options);

    }

/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

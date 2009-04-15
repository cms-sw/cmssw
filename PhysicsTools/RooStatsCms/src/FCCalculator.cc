// @(#)root/hist:$Id: FCCalculator.cc,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include <iostream>

#include "TTree.h"
#include "TFile.h"

#include "RooDataSet.h"
#include "RooFormulaVar.h"
#include "RooFitResult.h"
#include "RooGlobalFunc.h" // for RooFit::Extended()
#include "RooMinuit.h"
#include "RooAddPdf.h"

#include "PhysicsTools/RooStatsCms/interface/LikelihoodCalculator.h"

#include "PhysicsTools/RooStatsCms/interface/FCCalculator.h"



/*----------------------------------------------------------------------------*/

FCCalculator::FCCalculator(const char* name,
                           const char* title,
                           RooAbsPdf* model,
                           RooArgList* variables,
                           const char* var_to_scan,
                           double var_measured_value,
                           ConstrBlockArray* c_array):
    StatisticalMethod(name,title,true),
    m_model(model),
    m_variables(variables),
    m_var_to_scan(var_to_scan),
    m_measured_value(var_measured_value){

    // The penalties

    if (c_array!=NULL){
        m_NLL_terms=new RooArgList(c_array->getNLLterms());
        m_NLL_penalties=c_array->getNLLstring();
        }
    else
        m_NLL_terms=new RooArgList();
    }

/*----------------------------------------------------------------------------*/

FCCalculator::~FCCalculator(){

//     if (m_NLL_terms!=NULL)
//         delete m_NLL_terms;
// 
//     if (m_variables!=NULL)
//         delete m_variables;
    }

/*----------------------------------------------------------------------------*/

/**
The very heart of the method. Here the scans are carried out.
The idea is, for every toy:
 - Get the deltaNLL for the point to study
 - Get the deltaNLL between the point to study and the point corrseponding to 
   the measured value.
Therefore 2 families of R values for each point so to perform a study later on 
the behaviour of theh limit.

**/

void FCCalculator::calculate(unsigned int n_toys,
                             const char* rootfilename,
                             double studied_value){

    TFile ofile (rootfilename,"RECREATE");
    ofile.cd();

    double minimum_value;
    double nll_measured_value;
    double nll_studied_value;
    double nll_minimum_value;

    TTree FCdata("FCdata","Feldman Cousins toy-MC results");
    FCdata.Branch("measured_value",&m_measured_value,"measured_value/D");
    FCdata.Branch("studied_value",&studied_value,"studied_value/D");
    FCdata.Branch("minimum_value",&minimum_value,"minimum_value/D");
    FCdata.Branch("nll_measured_value",&nll_measured_value,"nll_measured_value/D");
    FCdata.Branch("nll_studied_value",&nll_studied_value,"nll_studied_value/D");
    FCdata.Branch("nll_minimum_value",&nll_minimum_value,"nll_minimum_value/D");

    /*
    Perform the toys
    For each one get the tree points: the studied one, the minimum and the one
    at the measured. In other words just three points of the likelihood scan.
    To get the minimum of the scan we just do a minimisation of the likelihood.
    To get the other two points we fix the value of the par to scan and we 
    minimise.
    */
    RooArgSet* vars = m_model->getVariables();

    RooRealVar* var=static_cast<RooRealVar*>
                          (vars->find(m_var_to_scan.Data()));

    assert(var!=NULL);

    RooDataSet* data;
    LikelihoodCalculator* lcalc;
    RooFormulaVar* nll;
    RooFitResult* result;
    RooFitResult* result_studied;

    m_save_params(vars);

    for (unsigned int toy_index=0;toy_index<n_toys;++toy_index){
        // Set the value of the n_sig to the point to study
        var->setVal(studied_value);

        data = m_model->generate(*m_variables,RooFit::Extended());

        // Build the NLL var
        lcalc = new LikelihoodCalculator (*m_model,
                                          *data);
        nll = lcalc->getNLL();

        // The nll value at the studied value
        var->setVal(studied_value);
        var->setConstant();
        RooMinuit minuit_studied(*nll);
        minuit_studied.migrad();
        result_studied = minuit_studied.save();
        nll_studied_value = nll->getVal();
        var->setConstant(false);
        m_restore_params(vars);

        // The nll_value at the measured value
        var->setVal(m_measured_value);
        nll_measured_value = nll->getVal();

        /*
        The minimum_value:
        set the value of the variable to its original value
        */
        var->setVal(studied_value);
        RooMinuit minuit(*nll);
        minuit.migrad();
        result = minuit.save();
        minimum_value = var->getVal();
        nll_minimum_value=nll->getVal();

        FCdata.Fill();

        m_restore_params(vars);

        if (lcalc!=NULL)
            delete lcalc;
        delete data;
        delete result;
        delete result_studied;
        }

    FCdata.Write();
    ofile.Close();

    }

/*----------------------------------------------------------------------------*/

void FCCalculator::print(const char* options){

    std::cout << "\n FCCalculator object: " << GetName() << "\n\n"
              << GetTitle() << std::endl
              << " - Model: ";
    m_model->Print(options);
    std::cout << " - Variables: ";
    m_variables->Print(options);
    std::cout << " - Variable to scan name: " << m_var_to_scan << std::endl
              << " - Measured value: " << m_measured_value << std::endl;

    }

/*----------------------------------------------------------------------------*/

void FCCalculator::m_save_params(RooArgSet* vars){
    std::cout << "In func\n";

    TString constr_name("Constraint");
    TString rrvar_name("RooRealVar");

    std::cout << "Model class is a " << m_model->ClassName() <<std::endl;
    TIterator* it = vars->createIterator();
    TObject* obj;

    obj=it->Next();
    while(obj!=NULL){
        if (rrvar_name==obj->ClassName()){
            m_params_vals.push_back((static_cast<RooRealVar*>(obj))->getVal());
            std::cout << "Class is a " << obj->ClassName() 
                      << " with value "<< (static_cast<RooRealVar*>(obj))->getVal()
                      << std::endl;
            }
        if (constr_name==obj->ClassName()){
            m_params_vals.push_back((static_cast<Constraint*>(obj))->getVal());
            std::cout << "Class is a " << obj->ClassName() 
                      << " with value "<< (static_cast<Constraint*>(obj))->getVal()
                      << std::endl;
            }
        obj=it->Next();
        }
    delete it;
    }


/*----------------------------------------------------------------------------*/

void FCCalculator::m_restore_params(RooArgSet* vars){
    TString constr_name("Constraint");
    TString rrvar_name("RooRealVar");

    std::cout << "Restoring parmas\n";
    TIterator* it = vars->createIterator();
    TObject* obj;
    int param_index=0;
    obj=it->Next();
    while(obj!=NULL){

        if (obj==NULL)
            break;
        if (rrvar_name==obj->ClassName()){
            (static_cast<RooRealVar*>(obj))->setVal(m_params_vals[param_index++]);
            std::cout << "Class is a " << obj->ClassName() 
                      << " with restored value "<< (static_cast<RooRealVar*>(obj))->getVal()
                      << std::endl;
            }
        else if (constr_name==obj->ClassName()){
            (static_cast<Constraint*>(obj))->setVal(m_params_vals[param_index++]);
            std::cout << "Class is a " << obj->ClassName() 
                      << " with restored value "<< (static_cast<Constraint*>(obj))->getVal()
                      << std::endl;
            }
        obj=it->Next();
        }
    }

/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

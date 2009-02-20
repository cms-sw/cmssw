// @(#)root/hist:$Id: PLScan.cc,v 1.1 2009/01/06 12:22:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008

#include "assert.h"
#include <iostream>

#include "RooRealVar.h"
#include "RooMinuit.h"
#include "RooFitResult.h"

#include "TIterator.h"

#include "PhysicsTools/RooStatsCms/interface/PLScan.h"

/// To build the cint dictionaries
//ClassImp(PLScan)

/*----------------------------------------------------------------------------*/

PLScan::PLScan(const char* name,
               const char* title,
               RooFormulaVar* nll,
               const char* varToscan,
               std::vector<double> points,
               double varGenVal):
    StatisticalMethod(name,title,true),
    m_nll(nll),
    m_scanned_parameter_name(varToscan),
    m_generated_value(varGenVal){

    setPoints(points);

    }

/*----------------------------------------------------------------------------*/

PLScan::PLScan(const char* name,
               const char* title,
               RooFormulaVar* nll,
               const char* varToscan,
               double scan_min,
               double scan_max,
               unsigned int npoints,
               double varGenVal):
    StatisticalMethod(name,title,true),
    m_nll(nll),
    m_scanned_parameter_name(varToscan),
    m_generated_value(varGenVal){
    setPoints(scan_min,scan_max,npoints);

    }


/*----------------------------------------------------------------------------*/

PLScan::PLScan(const char* name,
               const char* title,
               RooFormulaVar* nll,
               const char* varToscan,
               double varGenVal):
    StatisticalMethod(name,title,true),
    m_nll(nll),
    m_scanned_parameter_name(varToscan),
    m_generated_value(varGenVal){
    }

/*----------------------------------------------------------------------------*/

/**
Set the grid of the scan points from a vector.
**/

void PLScan::setPoints(std::vector<double> points){

    for (std::vector<double>::iterator it = points.begin(); 
         it!=points.end(); ++it)
        m_points_grid.push_back(*it);

    }

/*----------------------------------------------------------------------------*/

/**
Set the grid of the scan points from an interval and a points number.
**/

void PLScan::setPoints(double scan_min, double scan_max, unsigned int npoints){

    assert(scan_min<scan_max);

    double step=(scan_max-scan_min)/npoints;
    for (unsigned int i=0;i<npoints;++i)
        m_points_grid.push_back(step*i+scan_min);

    }

/*----------------------------------------------------------------------------*/

/**
Perform the likelihood scan. If the profile flag is true the NLL is minimised 
at each point. This is the default behaviour.
A PLScanResults object is returned.
**/

PLScanResults* PLScan::doScan(bool profile){

    // Acquire the parameter to scan_min
    RooRealVar *par;
    TIterator* par_it = m_nll->getVariables()->createIterator();
    par=(RooRealVar*) par_it->Next();
    bool found=false;
    while(par_it!=NULL){
        if (m_scanned_parameter_name.Contains(par->GetName())){
            found = true;
            break;
            }
        par=(RooRealVar*) par_it->Next();
        }

    if (not found){
        std::cout << "[PLScan::doScan] No parameter " 
                  << m_scanned_parameter_name 
                  << " is present. Aborting..";
        abort();
        }

    delete par_it;

    // Allocate the minuit object to minimise in case the NNL and a results ptr
    RooMinuit minuit(*m_nll);
    RooFitResult* fit_result;

    double original_value = par->getVal();

    // Begin of the scan
    for (std::vector<double>::iterator scan_value = m_points_grid.begin(); 
         scan_value!=m_points_grid.end(); ++scan_value){

        if (is_verbose())
            std::cout << "[PLScan::doScan] "
                      << " Scan value: " << *scan_value << " ...\n";
        // Fix the param value
        par->setVal(*scan_value);
        par->setConstant(true);

        // Minimise NLL if requested
        if (profile){
            minuit.migrad();
            fit_result = minuit.save();
            if (is_verbose())
                fit_result->Print("v");
            delete fit_result;
            }

        m_NLL_values.push_back(m_nll->getVal());

        // Release the variable
        par->setConstant(false);

        } //end of the scan!

    // Build the PLScanResults object
    TString name(GetName());
    name+="_result";

    TString title(GetTitle());
    title+="_result";

    PLScanResults* result=new PLScanResults(name,
                                            title,
                                            m_points_grid,
                                            m_NLL_values,
                                            m_scanned_parameter_name.Data(),
                                            m_generated_value);
    result->setVerbosity(false);

    par->setVal(original_value);

    return result;
    }

/*----------------------------------------------------------------------------*/

/**
Print info about the object.
**/
void PLScan::print(const char* options){

    std::cout << "\n PLScan object: \n\n"
              << " - Name:" << GetName() << std::endl
              << " - Title:" << GetTitle() << std::endl
              << " - Scanned points: " << m_points_grid.size() << std::endl
              << " - Generated value (default =  " << DUMMY_VALUE
                 << "):" << m_generated_value << std::endl
              << " - RooNllVar: \n";
    m_nll->Print("v");

    }

/*----------------------------------------------------------------------------*/







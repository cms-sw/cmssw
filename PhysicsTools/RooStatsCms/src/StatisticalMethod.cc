// @(#)root/hist:$Id: StatisticalMethod.cc,v 1.1 2008/06/05 20:05:39 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#include "StatisticalMethod.h"

/*----------------------------------------------------------------------------*/

StatisticalMethod::StatisticalMethod(const char* name,
                                     const char* title,
                                     bool verbosity):
    TNamed(name,title){
    setVerbosity(verbosity);
    }

/*----------------------------------------------------------------------------*/

void StatisticalMethod::setVerbosity(bool verbosity){
    m_verbose=verbosity;
    }

/*----------------------------------------------------------------------------*/

bool StatisticalMethod::is_verbose(){
    return m_verbose;
    }

/// To build the cint dictionaries
ClassImp(StatisticalMethod)

// @(#)root/hist:$Id: StatisticalMethod.cc,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"

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

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

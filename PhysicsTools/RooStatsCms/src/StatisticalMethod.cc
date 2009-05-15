// @(#)root/hist:$Id: StatisticalMethod.cc,v 1.3 2009/04/15 11:10:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#if (defined (STANDALONE) or defined (__CINT__) )
   #include "StatisticalMethod.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"
#endif

//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassImp(StatisticalMethod)
#endif
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

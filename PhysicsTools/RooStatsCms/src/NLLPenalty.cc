// @(#)root/hist:$Id: NLLPenalty.cc,v 1.1 2009/01/06 12:22:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#include "PhysicsTools/RooStatsCms/interface/NLLPenalty.h"

/*----------------------------------------------------------------------------*/

void NLLPenalty::setVerbosity(bool verbosity){
    m_verbose=verbosity;
    }

/*----------------------------------------------------------------------------*/

bool NLLPenalty::is_verbose(){
    return m_verbose;
    }

/*----------------------------------------------------------------------------*/

/// To build the cint dictionaries
//ClassImp(NLLPenalty)

/*----------------------------------------------------------------------------*/

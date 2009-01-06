// @(#)root/hist:$Id: NLLPenalty.cc,v 1.1 2008/06/06 17:37:48 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#include "NLLPenalty.h"

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
ClassImp(NLLPenalty)

/*----------------------------------------------------------------------------*/

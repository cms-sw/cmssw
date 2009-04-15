// @(#)root/hist:$Id: NLLPenalty.cc,v 1.1.1.1 2009/04/15 08:40:01 dpiparo Exp $
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


/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

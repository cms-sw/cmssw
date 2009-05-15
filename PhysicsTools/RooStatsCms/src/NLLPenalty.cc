// @(#)root/hist:$Id: NLLPenalty.cc,v 1.3 2009/04/15 11:10:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#if (defined (STANDALONE) or defined (__CINT__) )
   #include "NLLPenalty.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/NLLPenalty.h"
#endif

//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassImp(NLLPenalty)
#endif
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

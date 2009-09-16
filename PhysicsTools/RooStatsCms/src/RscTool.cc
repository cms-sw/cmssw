// @(#)root/hist:$Id: RscTool.cc,v 1.4 2009/05/15 09:55:59 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#if (defined (STANDALONE) or defined (__CINT__) )
   #include "RscTool.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/RscTool.h"
#endif


RscTool::RscTool(const char* name,const  char* title, bool verbosity)
    :TNamed(name,title){
    setVerbosity(true);
    }

//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassImp(RscTool)
#endif
/*----------------------------------------------------------------------------*/

void RscTool::setVerbosity(bool verbosity){
    m_verbose=verbosity;
    }

/*----------------------------------------------------------------------------*/

bool RscTool::is_verbose(){
    return m_verbose;
    }

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

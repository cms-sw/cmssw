// @(#)root/hist:$Id: RscTool.cc,v 1.3 2009/04/15 11:10:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#if (defined (STANDALONE) or defined (__CINT__) )
   #include "RscTool.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/RscTool.h"
#endif


RscTool::RscTool(char* name, char* title, bool verbosity)
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

// @(#)root/hist:$Id: RscTool.cc,v 1.1 2009/01/06 12:22:44 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch   01/06/2008


#include "PhysicsTools/RooStatsCms/interface/RscTool.h"


RscTool::RscTool(char* name, char* title, bool verbosity)
    :TNamed(name,title){
    setVerbosity(true);
    }

/*----------------------------------------------------------------------------*/

void RscTool::setVerbosity(bool verbosity){
    m_verbose=verbosity;
    }

/*----------------------------------------------------------------------------*/

bool RscTool::is_verbose(){
    return m_verbose;
    }

/// To build the cint dictionaries
//ClassImp(RscTool)

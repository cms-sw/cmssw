// @(#)root/hist:$Id: RscTool.h,v 1.1 2009/01/06 12:18:38 dpiparo Exp $
// Author: Danilo.Piparo@cern.ch, Gregory.Schott@cern.ch   05/04/2008

/// RscTool : The mother class of the RooStatsCms Tools

/**
\class RscTool
$Revision: 1.1 $
$Date: 2009/01/06 12:18:38 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott (grgory.schott<at>cern.ch) - Universitaet Karlsruhe 
The mother class of the RooStatsCms Tools.
**/

#ifndef __RscTool__
#define __RscTool__

#include "TNamed.h"

class RscTool : public TNamed  {

  public:

    RscTool(char* name, char* title, bool verbosity=true);

    /// Set the verbosity
    void setVerbosity(bool verbosity);

    /// get the verbosity
    bool is_verbose();

  private:

    /// Verbosity flag
    bool m_verbose;

    // For Cint
    //ClassDef(RscTool,1) 

};

#endif

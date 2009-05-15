/// StatisticalMethod: the base class for the statistical methods.

/**
\class StatisticalMethod
$Revision: 1.3 $
$Date: 2009/04/15 11:10:45 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class provides the base for all the statistical methods.
**/

#ifndef __StatisticalMethod__
#define __StatisticalMethod__

#include "TNamed.h"

class StatisticalMethod : public TNamed {

  public:

    /// Constructor
    StatisticalMethod(const char* name,const  char* title, bool verbosity=true);

    /// Set the verbosity
    void setVerbosity(bool verbosity);

    /// get the verbosity
    bool is_verbose();

    /// Print the relevant information
    virtual void print(const char* options="") = 0;

  private:

    /// Verbosity flag
    bool m_verbose;

//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(StatisticalMethod,1)
#endif
 };

#endif

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

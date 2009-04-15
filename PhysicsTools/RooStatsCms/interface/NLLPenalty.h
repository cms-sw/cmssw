/// NLLPenalty: base class for the blocks of correlations and constraints

/**
\class NLLPenalty
$Revision: 1.1.1.1 $
$Date: 2009/04/15 08:40:01 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class is the base for the blocks of correlations and constraints
**/

#ifndef __NLLPenalty__
#define __NLLPenalty__


#include <iostream>

#include "TString.h"

#include "RooArgList.h"

# include "TRandom3.h"


#define SIG_KEYWORD "sig"


class NLLPenalty {

  public:

    /// Set the verbosity
    void setVerbosity(bool verbosity);

    /// get the verbosity
    bool is_verbose();

    /// Print the relevant information
    virtual void print(const char* options="") = 0;

    /// Fluctuate parameters
    virtual void fluctuate() = 0;

    /// Restore parameters original values
    virtual void restore() = 0;

    /// Fix the constraints to their nominal value
    virtual void setFixed(bool fix) = 0;

    /// Get NLL string
    virtual TString getNLLstring() = 0;

    /// Get NLL string only for background (keyword _sig)
    virtual TString getBkgNLLstring() = 0;

    /// Get NLL terms in a list
    virtual RooArgList getNLLterms() = 0;

    /// Get NLL terms in a list only for background (keyword _sig)
    virtual RooArgList getBkgNLLterms() = 0;

    ///// Virtual destructor
    //virtual ~NLLPenalty() = 0;

    /// The random generator
    TRandom3 random_generator;


  private:

    /// Verbosity flag
    bool m_verbose;

 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

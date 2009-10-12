/// ConstrBlockArray: an array of correlations and constraints

/**
\class ConstrBlockArray
$Revision: 1.1.1.1 $
$Date: 2009/04/15 08:40:01 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class is a container for the contraints and correlations.
**/

#ifndef __ConstrBlockArray__
#define __ConstrBlockArray__


#include "TNamed.h"
#include "TString.h"

#include "PhysicsTools/RooStatsCms/interface/Constraint.h"

const int MAX_LENGHT=50;

class ConstrBlockArray : public TNamed,public NLLPenalty {

  public:

    /// Constructor
    ConstrBlockArray (const char* name,
                      const char* title,
                      NLLPenalty* penalty1=NULL,
                      NLLPenalty* penalty2=NULL,
                      NLLPenalty* penalty3=NULL,
                      NLLPenalty* penalty4=NULL,
                      NLLPenalty* penalty5=NULL,
                      NLLPenalty* penalty6=NULL,
                      NLLPenalty* penalty7=NULL,
                      NLLPenalty* penalty8=NULL);

    /// Add a penalty
    void add(NLLPenalty* penalty);

    /// Print the relevant information
    void print(const char* options="");

    /// Fluctuate parameters
    void fluctuate();

    /// Restore parameters original values
    void restore();

    /// Fix the constraints to their nominal value
    void setFixed(bool fix);

    /// Get NLL string
    TString getNLLstring();

    /// Get NLL string for Bkg
    TString getBkgNLLstring();

    /// Get NLL terms in a list
    RooArgList getNLLterms();

    /// Get NLL terms in a list
    RooArgList getBkgNLLterms();

    /// Set the array to own the content or not
    void ownContent(bool own){m_owns_content=own;}

    /// Get block
    NLLPenalty* getBlock(int index);

    /// Get size
    int getSize(){return m_size;}

    /// Destructor
    ~ConstrBlockArray();

  private:

    /// Here the penalties are stored
    NLLPenalty* m_penalties[MAX_LENGHT];

    /// Here is the counter for the penalties
    int m_size;

    /// Own flag
    bool m_owns_content;

 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

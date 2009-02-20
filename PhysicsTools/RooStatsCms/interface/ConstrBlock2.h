/// ConstrBlock2: the block of 2 correlated constraints

/**
\class ConstrBlock2
$Revision: 1.1 $
$Date: 2009/01/06 12:18:36 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class represents the block of two correlated constraints. 
NOTE: only gaussian constraints are possible ate the moment.
**/

#ifndef __ConstrBlock2__
#define __ConstrBlock2__


#include "TNamed.h"
#include "TString.h"

#include "RooArgList.h"

#include "PhysicsTools/RooStatsCms/interface/Constraint.h"

class ConstrBlock2 : public TNamed, public NLLPenalty {

  public:

    /// Constructor
    ConstrBlock2 (const char* name,
                const char* title,
                double corr12,
                Constraint* constr1,
                Constraint* constr2);

    /// Print the relevant information
    void print(const char* options="");

    /// Fluctuate parameters
    void fluctuate();

    /// Restore parameters original values
    void restore();

    /// Get NLL string
    TString getNLLstring(){return m_NLL_string;};

    /// Get NLL string for background
    TString getBkgNLLstring();

    /// Get NLL terms in a list
    RooArgList getNLLterms(){return *m_parameters;};

    /// Get NLL terms in a list for background
    RooArgList getBkgNLLterms();

    /// Fix the constraints to their nominal value
    void setFixed(bool fix);

    /// Destructor
    ~ConstrBlock2();


  private:

    /// Correlation coefficient
    RooRealVar* m_corr;

    /// Constraints array
    RooArgList* m_constr_list;

    /// List of the parameters
    RooArgList* m_parameters;

    /// Penalty NLL string
    TString m_NLL_string;

    // For Cint
    //ClassDef(ConstrBlock2,1) 
 };

#endif

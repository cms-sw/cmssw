/// ConstrBlock3: the block of 2 correlated constraints

/**
\class ConstrBlock3
$Revision: 1.1.1.1 $
$Date: 2009/04/15 08:40:01 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class represents the block of three correlated constraints. 
NOTE: only gaussian constraints are possible ate the moment.
**/

#ifndef __ConstrBlock3__
#define __ConstrBlock3__


#include "TNamed.h"
#include "TString.h"

#include "RooArgList.h"

#include "PhysicsTools/RooStatsCms/interface/Constraint.h"

class ConstrBlock3 : public TNamed, public NLLPenalty {

  public:

    /// Constructor
    ConstrBlock3 (const char* name,
                const char* title,
                double corr12,
                double corr13,
                double corr23,
                Constraint* constr1,
                Constraint* constr2,
                Constraint* constr3);

    /// Print the relevant information
    void print(const char* options="");

    /// Fluctuate parameters
    void fluctuate();

    /// Restore parameters original values
    void restore();

    /// Get NLL string
    TString getNLLstring(){return m_NLL_string;};

    /// Get NLL string for bkg
    TString getBkgNLLstring(){return m_Bkg_NLL_string;};

    /// Get NLL terms in a list
    RooArgList getNLLterms(){return *m_parameters;};

    /// Get NLL terms in a list for bkg
    RooArgList getBkgNLLterms(){return *m_Bkg_parameters;};

    /// Fix the constraints to their nominal value
    void setFixed(bool fix);

    /// Destructor
    ~ConstrBlock3();


  private:

    /// Correlation coefficient
    RooRealVar* m_corr[3];

    /// Constraints array
    RooArgList* m_constr_list;

    /// List of the parameters
    RooArgList* m_parameters;

    /// List of the parameters for the Bkg
    RooArgList* m_Bkg_parameters;

    /// Penalty NLL string
    TString m_NLL_string;

    /// Penalty NLL string for the background
    TString m_Bkg_NLL_string;

    /// Get the parameters of the background
    void m_getBkgConstraints(Constraint** c,int* n, double* corr);

 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

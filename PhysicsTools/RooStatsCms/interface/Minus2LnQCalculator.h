/// Minus2LnQCalculator: an easy way to calculate 2lnQ

/**
\class Minus2LnQCalculator
$Revision: 1.1 $
$Date: 2009/01/06 12:18:37 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class eases the calculation of a likelihood ratios taking into account 
additional terms (correlations and constraints) and minimisation.
NOTE: in future a multithreaded version is foreseen.
**/

#ifndef __Minus2LnQCalculator__
#define __Minus2LnQCalculator__

#include "PhysicsTools/RooStatsCms/interface/LikelihoodCalculator.h"

#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooArgList.h"



class Minus2LnQCalculator : public TNamed {

  public:

    /// Constructor
    Minus2LnQCalculator (RooAbsPdf& sb_model_pdf,
                         RooAbsPdf& b_model_pdf,
                         RooAbsData& dataset);

    /// Constructor with penality terms
    Minus2LnQCalculator (RooAbsPdf& sb_model_pdf,
                         RooAbsPdf& b_model_pdf,
                         TString& formula,
                         RooArgList terms,
                         RooAbsData& dataset);

    /// Constructor with penality terms, different from sb and b models
    Minus2LnQCalculator (RooAbsPdf& sb_model_pdf,
                         RooAbsPdf& b_model_pdf,
                         TString& sb_formula,
                         RooArgList sb_terms,
                         TString& b_formula,
                         RooArgList b_terms,
                         RooAbsData& dataset);

    /// Destructor
    ~Minus2LnQCalculator ();

    void free(int i);

    /// Get the value
    double getValue(bool minimise=true);

    /// Get the value of sqrt(2lnQ)
    double getSqrtValue(bool minimise=true);

    /// Set the verbosity
    void setVerbosity (bool verbose){m_verbose=verbose;};

  private:

    /// The likelihood calculators
    LikelihoodCalculator* m_Lcalcs[2];

    /// Operations common to the constructors with penalties
    void m_init_with_penalties(RooAbsPdf& sb_model_pdf,
                               RooAbsPdf& b_model_pdf,
                               TString& sb_formula,
                               RooArgList sb_terms,
                               TString& b_formula,
                               RooArgList b_terms,
                               RooAbsData& dataset);

    /// The verbosity flag
    bool m_verbose;

    // For Cint
    //ClassDef(Minus2LnQCalculator,1) //Calculate Limit using the CLs method (A. Read)
 };

#endif

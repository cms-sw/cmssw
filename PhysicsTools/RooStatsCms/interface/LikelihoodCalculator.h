/// LikelihoodCalculator: an easy way to calculate a likelihood

/**
\class LikelihoodCalculator
$Revision: 1.1 $
$Date: 2009/01/06 12:18:36 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class eases the calculation of a likelihood taking into account additional 
terms and minimisation.
**/

#ifndef __LikelihoodCalculator__
#define __LikelihoodCalculator__

#include <vector>

#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooNLLVar.h"
#include "RooArgList.h"



class LikelihoodCalculator : public TNamed {

  public:

    /// Constructor
    LikelihoodCalculator (RooAbsPdf& model_pdf, 
                          RooAbsData& dataset);

    /// Constructor with penality terms
    LikelihoodCalculator (RooAbsPdf& model_pdf, 
                          RooAbsData& dataset,
                          TString& penalty_formula,
                          RooArgList& penalty_terms);

    /// Destructor
    ~LikelihoodCalculator ();

    /// Get the NLL value
    double getValue(bool minimise=true);

    /// Set the verbosity
    void setVerbosity (bool verbose){m_verbose=verbose;};

    ///Get the NLL
    RooFormulaVar* getNLL(){return m_nll_constr;};

  private:

    /// The internal base negative Likelihood function
    RooNLLVar* m_base_nll;

    /// The internal Likelihood function containing also the penalty terms
    RooFormulaVar* m_nll_constr;

    /// Restore the original value of the params
    void m_restore_params_values (RooFormulaVar* nll);

    /// Store the original values of the params
    void m_save_params_values (RooFormulaVar* nll);

    /// Buffer for the params values
    std::vector<double> m_original_var_values;

    /// Verbosity flag
    bool m_verbose;

    // For Cint
    //ClassDef(LikelihoodCalculator,1) //Calculate Limit using the CLs method (A. Read)
 };

#endif

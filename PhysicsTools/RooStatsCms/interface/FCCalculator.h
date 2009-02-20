/// FCCalculator: The Feldman-Cousins unified approach to the Classical Statistical Analysis of Small Signals

/**
\class FCCalculator
$Revision: 1.1 $
$Date: 2009/01/06 12:18:36 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

The full Neyman construction was introduced to HEP by Feldman and Cousins in 
<a href="http://prola.aps.org/abstract/PRD/v57/i7/p3873_1"> this </a> paper. 
The test statistic is the likelihood ratio Q(s) = L(s+b) / L(s_hat+b) where 
s_hat is the physically allowed mean s that maximizes the Likelihood 
L(s_hat + b). To construct an acceptance 68% interval in the number of observed 
events, [n1, n2], one is using Q as an ordering rule, i.e. 
/f$\sum^{n2}_{n1} p(n|s,b) \geq 68% /f$
where only terms with decreasing order of Q(n) are included in the sum, till 
the sum exceeds the 68% confidence. When 
/f$n_{0}/f$ 
events are observed, one is using this constructed Neyman belt to derive a 
confidence interval, which, depending on the observation,
might be a one-sided or a two-sided interval. This method is therefore called 
the unified method, because it avoids a flip-flop of the inference (i.e. one 
decides to fliip from a limit to an interval if the result is significant 
enough...).
The difficulty with this approach is that an experiment with higher expected 
background which observes no events might set a better upper limit than an 
experiment with lower or no expected background. This would never occur with 
the CLs method (See LimitCalculator). 
Another difficulty is that this approach does not incorporate a treatment of 
nuisance parameters. However, it can either be plugged in "by hand", using the 
hybrid <a href="http://prola.aps.org/abstract/PRD/v67/i1/e012002">Cousins and 
Highland method</a> or a Neyman construction can be performed.

This class allows to perform easily the toys to obtain the distribution of Q 
(in the code the lnQ is actually calculated) from which we can get the CL 
"points" to superimpose to the likelihood scan.
The procedure with which these Q distributions are calculated is rather heavy:
the design of the implementation of the Feldman Cousins method is therefore 
tailored for an execution on a batch system.

The main idea is that the FCCalculator instances produce rootfiles containing 
the trees with the relevant quantities. Thus the FCResults collects these 
rootfiles and is able to calculate the limits and produce the typical 
Feldman-Cousins plots that sobstitute the well known straight lines cutting 
the likelihood scan.
Out of the FCResults instance we can get the TGraphErrors with the FC points 
representing the CL. It is then possible to add them to a PLScanResults object 
to then easily calculate limits or to a PLScanPlot to obtain the final plot 
- FIXME here a sample FC plot.

For more information the procedure with which the toy Montecarlo experiments 
to obtain the Q distributions can be obtained, please see the /scripts 
directory where a template infrastructure working with the CERN batch system is 
present as an example.

**/

#ifndef __FCCalculator__
#define __FCCalculator__

#include <vector>

#include "TString.h"

#include "RooAbsPdf.h"
#include "RooArgList.h"

#include "PhysicsTools/RooStatsCms/interface/ConstrBlockArray.h"

#include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"


class FCCalculator : public StatisticalMethod {

  public:

    /// Constructor
    FCCalculator(const char* name,
                 const char* title,
                 RooAbsPdf* model,
                 RooArgList* variables,
                 const char* var_to_scan,
                 double var_measured_value,
                 ConstrBlockArray* c_array=NULL); // Constructor with pdf models

    /// Destructor
    ~FCCalculator();

    /// Calculate limit with unbinned data
    void calculate(unsigned int n_toys,
                   const char* rootfilename,
                   double studied_value);

    /// Print relevant information
    void print(const char* options="");

  private:

    /// The model
    RooAbsPdf* m_model;

    /// The variables
    RooArgList* m_variables;

    /// The name of the variable to scan
    TString m_var_to_scan;

    /// The measured value of the var 
    double m_measured_value;

    /// The NLL penalties
    TString m_NLL_penalties;

    /// The NLL penalties terms
    RooArgList* m_NLL_terms;

    /// Save params values
    void m_save_params(RooArgSet* vars);

    /// Restore params values
    void m_restore_params(RooArgSet* vars);

    /// Parameters values
    std::vector<double> m_params_vals;

    // For Cint
    //ClassDef(FCCalculator,1) //Calculate Limit using the Feldman Cousins Method
 };

#endif

/// LimitCalculator: A.Read's Method for confidence levels calculations

/**
\class LimitCalculator
$Revision: 1.1.1.1 $
$Date: 2009/04/15 08:40:01 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

The class is born from the need to have an implementation of the CLs 
method that could take advantage from the RooFit Package.
The basic idea is the following: 
- Instantiate an object specifying a signal+background model, a background model and a dataset.
- Perform toy MC experiments to know the distributions of -2lnQ 
- Calculate the CLsb and CLs values as "integrals" of these distributions.

The class allows the user to input models as RooAbsPdf or TH1 object 
pointers (the pdfs must be "extended": for more information please refer to 
http://roofit.sourceforge.net). The dataset can be entered as a 
RooTreeData or TH1 object pointer. 

Unlike the TLimit Class a complete MC generation is performed at each step 
and not a simple Poisson fluctuation of the contents of the bins.
Another innovation is the treatment of the nuisance parameters. The user 
can input in the constructor nuisance parameters and , optionally, the 
penalty terms of the loglikelihood linked to them.
Optionally the user can select the option ProfileLikelihood. In this case 
the likelihoods that appear in the Q ratio are profiled, i.e. the
likelihood is the maximum possible with respect to the nuisance parameters. 
Therefore to obtain a -2lnQ value the class performs 2 fits.
To include the information that we have about the nuisance parameters must be 
objects of the NuisanceParameter class.

The result of the calculations is returned as a LimitResults object pointer.

see also the following interesting references:
- Alex Read, "Presentation of search results: the CLs technique" Journal of Physics G: Nucl. Part. Phys. 28 2693-2704 (2002). http://www.iop.org/EJ/abstract/0954-3899/28/10/313/

- Alex Read, "Modified Frequentist Analysis of Search Results (The CLs Method)" CERN 2000-005 (30 May 2000)

- V. Bartsch, G.Quast, "Expected signal observability at future experiments" CMS NOTE 2005/004

- http://root.cern.ch/root/html/src/TLimit.html
**/

#ifndef __LimitCalculator__
#define __LimitCalculator__

#include <vector>

#include "TString.h"

#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooArgList.h"

#include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"
#include "PhysicsTools/RooStatsCms/interface/ConstrBlockArray.h"
#include "PhysicsTools/RooStatsCms/interface/LimitResults.h"


class LimitCalculator : public StatisticalMethod {

  public:

    /// Constructor
    LimitCalculator(const char* name,
                    const char* title,
                    RooAbsPdf* sb_model,
                    RooAbsPdf* b_model,
                    RooArgList* variables,
                    ConstrBlockArray* c_array=NULL); // Constructor with pdf models

    /// Destructor
    ~LimitCalculator();

    /// Calculate limit with unbinned data
    LimitResults* calculate(RooTreeData* data,
                            unsigned int n_toys,
                            bool fluctuate=false); //calculate limit with unbinned data


    /// Calculate limit without data. Set the value of -2lnQ later
    LimitResults* calculate(unsigned int n_toys,
                            bool fluctuate=false); //calculate limit with unbinned data


    /// Calculate limit with binned data
    LimitResults* calculate(TH1* data,
                            unsigned int n_toys,
                            bool fluctuate=false); //calculate limit with binned data

    /// Print relevant information
    void print(const char* options="");

  private:

    /// Do the mc experiments
    void m_do_toys(std::vector<float>& b_vals,
                   std::vector<float>& sb_vals,
                   unsigned int n_toys,
                   bool fluctuate=false);

    /// The pdf of the signal+background model
    RooAbsPdf* m_sb_model;

    /// The pdf of the background model
    RooAbsPdf* m_b_model;

    /// Collection of the variables of the model
    RooArgList* m_variables;

    /// Penalty string of the likelihood
    TString m_NLL_string;

    /// Penalty terms for the likelihood
    RooArgList m_NLL_terms;

    /// Penalty string of the likelihood, for bkg
    TString m_Bkg_NLL_string;

    /// Penalty terms for the likelihood, for bkg
    RooArgList m_Bkg_NLL_terms;

    /// The Array of the constraints
    ConstrBlockArray* m_c_array;

    /// Flag for the constraint fluctuation at each MC experiment
    bool m_fluctuate_constr;

    /// -2lnQ value on the data
    float m_m2lnQ_data;

 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

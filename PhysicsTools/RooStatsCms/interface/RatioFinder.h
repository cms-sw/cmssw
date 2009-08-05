/// RatioFinder: find the excluded production cross section

/**
\class RatioFinder
$Revision: 1.8 $
$Date: 2009/06/15 08:56:05 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

Find the production cross section to exclude at a fixed confidence level.
The idea is to have the signal yield expressed as a RooFormulaVar where the 
term called "ratio" appears.
**/

#ifndef __RatioFinder__
#define __RatioFinder__

#include "RooAbsPdf.h"
#include "RooArgList.h"

#include "TObjArray.h"

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "StatisticalMethod.h"
   #include "ConstrBlockArray.h"
   #include "LimitCalculator.h"
   #include "RatioFinderResults.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"
   #include "PhysicsTools/RooStatsCms/interface/ConstrBlockArray.h"
   #include "PhysicsTools/RooStatsCms/interface/LimitCalculator.h"
   #include "PhysicsTools/RooStatsCms/interface/RatioFinderResults.h"
#endif




class RatioFinder : public StatisticalMethod {

  public:

    /// Constructor
    RatioFinder(const char* name,
                const char* title,
                RooAbsPdf* sb_model,
                RooAbsPdf* b_model,
                const char* var_name,
                RooArgList variables,
                ConstrBlockArray* c_array=NULL); // Constructor with pdf models

    /// Find Ratio
    RatioFinderResults* findRatio(unsigned int n_toys,
                                  double init_lower_ratio,
                                  double init_upper_ratio,
                                  double n_sigma=0.,
                                  double CLs_level=0.05,
                                  double delta_ratios_min=0.5);

    /// Destructor
    ~RatioFinder();

    /// Print relevant information about the object instance
    void print(const char* options="");

    /// Set the number of bins
    void setNbins(int nbins){m_nbins=nbins;}

    /// Get the number of bins
    int getNbins(){return m_nbins;}

    /// Switch dumping intermediate results to pngs on or off
    void setDumpPlots(bool dump) { m_dump_plots = dump; }

    /// Get whether or not intermediate results are dumped to pngs
    bool getDumpPlots() { return m_dump_plots; }

    /// Save the intermediate results
    void saveIntermediateResultsIn(TObjArray *a) { m_results = a; }

  private:

    /// The pdf of the signal+background model
    RooAbsPdf* m_sb_model;

    /// The pdf of the background model
    RooAbsPdf* m_b_model;

    /// The Array of the constraints
    ConstrBlockArray* m_c_array;

    /// Switch to toggle dumping intermediate results to pngs
    bool m_dump_plots;

    /// The flag to decide if it is a lumi study
    bool m_is_lumi;

    /// The maximum number of attempts in the Ratio finding
    int m_max_attempts;

    /// The number of bins in the -2lnQ plot
    int m_nbins;

    /// Array to store intermediate results
    TObjArray *m_results;

    /// Get the Cls value
    double m_get_CLs(double ratio,
                     unsigned int n_toys,
                     double& m2lnQ,
                     double n_sigma);

    /// Get the result of LimitCalculator
    LimitResults* m_get_LimitResults(unsigned int n_toys);

    /// Make wheighted average
    double m_weighted_average(double h_val,
                              double h_weight,
                              double l_val,
                              double l_weight);

    /// Collection of the variables of the model
    RooArgList m_variables;

    /// The epsilon between the 2 ratios
    double m_delta_ratios_min;

    /// The ratio variable
    RooRealVar* m_ratio;


//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(RatioFinder,1)
#endif
 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

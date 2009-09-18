/// FCResults: The Feldman-Cousins unified approach to the Classical Statistical Analysis of Small Signals

/**
\class FCResults
$Revision: 1.1.1.1 $
$Date: 2009/04/15 08:40:01 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

In the following the procedure to collect the results of the FCCalculator and 
get a plot is described. For a satisfactory documentation of the statistical 
method, please see FCCalculator documentation.

This class collects the rootfiles produced by the FCCalculator according to the 
wildcard given in the constructor (the very wildard is passed to the 
constructor of an internal TChain).
The studied points are fetched examining the TChain on the fly. This has two 
big advantages:
 - It's possible to monitor the evolution of the MC generation, without waiting 
   the end of the jobs.
 - We can add points or increase the statistic at any time.


**/

#ifndef __FCResults__
#define __FCResults__

#include <vector>

#include "TString.h"
#include "TH1F.h"
#include "TGraphErrors.h"

#include "RooAbsPdf.h"
#include "RooArgList.h"

#include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"

#define MAX_SCAN_POINTS 100

class FCResults : public StatisticalMethod {

  public:

    /// Constructor
    FCResults(const char* name,
              const char* title,
              const char* rootfiles_wildcard);

    /// Destructor
    ~FCResults();
/*
    /// Get the upper limit
    double getUL(double ConfidenceLevel);

    /// Get the lower limit
    double getLL(double ConfidenceLevel);

    /// Get the plot object
    FCPlot* getPlot(char* name="",char* title="");
*/
    /// Get the x grid
    std::vector<double> getXgrid(){return m_x_grid;};

    /// Get R histo using minimum NLL
    TH1F* getRhistoMinimum(double grid_index) {return m_x_RminHisto_map[grid_index];};

    /// Get R histo using measured NLL
    TH1F* getRhistoMeasured(double grid_index){return m_x_RmeasHisto_map[grid_index];};

    /// Get CL points using minimum NLL
    TGraphErrors* getCLpoints(double CL);

    /// Get CL points using measured NLL
    TGraphErrors* getCLpointsMeasured(double CL);

    /// Print relevant information
    void print(const char* options="");

  private:

    /// The grid of points studied
    std::vector<double> m_x_grid;

    /// The Rmin histograms
    //TH1F* m_Rmin_histos[MAX_SCAN_POINTS];

    /// The Rmeas histograms
    //TH1F* m_Rmeas_histos[MAX_SCAN_POINTS];

    /// The map x - histo for min values
    std::map <double, TH1F*> m_x_RminHisto_map;

    /// The map x - histo for meas values
    std::map <double, TH1F*> m_x_RmeasHisto_map;


    /// The cache map for the UL values: CL - Upper Limit
    std::map<double,double> m_CL_UL_map;

    /// The cache map for the LL values: CL - Lower Limit
    std::map<double,double> m_CL_LL_map;

    /// Check if an alement is in a vector
    int contains(std::vector<double>& vec, double val);

 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

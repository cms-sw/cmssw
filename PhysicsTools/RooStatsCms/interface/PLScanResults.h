/// PLScanResults: the result of a Likelihood scan

/**
\class PLScanResults
$Revision: 1.1 $
$Date: 2009/01/06 12:18:37 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

The results of the likelihood scan.

**/

#ifndef __PLScanResults__
#define __PLScanResults__

#include <vector>
#include <map>
#include "assert.h"

#include "TString.h"
#include "TGraph.h"

#include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"
#include "PhysicsTools/RooStatsCms/interface/PLScanPlot.h"

#define DUMMY_VALUE 12345e-50

class PLScanResults : public StatisticalMethod {

  public:

    /// Constructor
    PLScanResults(const char* name,
                  const char* title,
                  std::vector<double> points_grid,
                  std::vector<double> NLL_values,
                  const char* scanned_var_name,
                  double genval=DUMMY_VALUE);

    /// Constructor
    PLScanResults();

    /// Get CL from a deltaNLL value
    double getDeltaNLLfromCL(double CL);

    /// Get CL from a deltaNLL value in case of an upper limit
    double getDeltaNLLfromCL_limit(double CL);

    /// Get a deltaNLL from a CL value
    double getCLfromDeltaNLL(double DeltaNLL);

    // Get a deltaNLL from a CL value in case of an upper limit
    //double getCLfromDeltaNLL_limit(double DeltaNLL);

    /// Add the results of another scan
    void add(PLScanResults results);

    /// Get the upper limit
    double getUL(double deltaNLL);

    /// Get the upper limit using Feldman-Cousins Graph
    double getUL(TGraphErrors* FC_graph, double x_min=0, double x_max=0);

    /// Get the lower limit
    double getLL(double deltaNLL);

    /// Get the lower limit using Feldman-Cousins Graph
    double getLL(TGraphErrors* FC_graph, double x_min=0, double x_max=0);

    /// Get the plot object
    PLScanPlot* getPlot(const char* name="",const char* title="");

//    /// Print the plot of the scan and the limit on a file
//    void printToFile(char* filename);

    /// Print the information about the object
    void print(const char* options);

    /// Get the scan minimum
    const double* getScanMinimum(){return (const double* )m_scan_minimum;};

    /// Check the coverage
    bool isCovering(double CL);

    /// Get the scan range minimum
    double getScanRangeMin(){return m_scan_range_min;};

    /// Get the scan range maximum
    double getScanRangeMax(){return m_scan_range_max;};

    /// Get the scan values
    std::vector<double> getScanValues(){return m_NLL_shifted_values;};

    /// Get the unshifted scan values
    std::vector<double> getUnshiftedScanValues(){return m_NLL_values;};

  private:

    /// Find the scan extreme points
    void m_fill_scan_range_extremes();

    /// Scan minimum value
    double m_scan_range_min;

    /// Scan maximum value
    double m_scan_range_max;

    /// Generated value buffer:written at construction time
    double m_generated_value;

    /// Init values
    void m_add_scan_values(std::vector<double> points_grid,
                           std::vector<double> NLL_values);

    /// The scan minimum 
    double m_scan_minimum[2];

    /// The grid points
    std::vector<double> m_points_grid;

    /// The NLL values at the points
    std::vector<double> m_NLL_values;

    /// The NLL values at the points
    std::vector<double> m_NLL_shifted_values;

    /// Fill the NLL shifted values
    void m_fill_NLL_shifted_values();

    /// The cache map for the UL values: CL - Upper Limit
    std::map<double,double> m_CL_UL_map;

    /// The cache map for the LL values: CL - Lower Limit
    std::map<double,double> m_CL_LL_map;

    /// Find the scan minimum
    void m_find_scan_minimum();

    /// Obtain parabola coefficients
    void m_parabola(double* x, double* y,
                    double& a, double& b, double& c);

    /// Obtain pol1 coefficients
    void m_pol1(double* p1, double* p2,
                double& m, double& q);

    ///Intersect orizzontal line with scan, if possible
    bool m_intersect(TF1* function,
                     double& intersection_asc,
                     double range_min,
                     double range_max);

    /// The stored name of the scanned parameter
    TString m_scanned_parameter_name;

    // For Cint
    //ClassDef(PLScanResults,1)
 };

#endif

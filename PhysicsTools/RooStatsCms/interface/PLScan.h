/// PLScan: Likelihood scan, profiling the likelihood

/**
\class PLScan
$Revision: 1.1 $
$Date: 2009/01/06 12:18:37 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

Implements the scan of the likelihood with respect to one parameter.
In the constructor the user imputs:
 - The name off the object
 - The title of the object
 - The Negative Log Likelihood
 - The name of trhe variable to scan
 - The points part of the scan (as a vector or as an interval+density)
 - The original value of the scanned parameter for coverage studies

The object returned from the doScan method is a PLScanResults.

**/

#ifndef __PLScan__
#define __PLScan__

#include <vector>

#include "TString.h"

#include "RooFormulaVar.h"

#include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"
#include "PhysicsTools/RooStatsCms/interface/PLScanResults.h"

class PLScan : public StatisticalMethod {

  public:

    /// Constructor with fixed points grid
    PLScan(const char* name,
           const char* title,
           RooFormulaVar* nll,
           const char* varToscan,
           std::vector<double> points,
           double varGenVal=DUMMY_VALUE);

    /// Constructor with automatic grid calculation
    PLScan(const char* name,
           const char* title,
           RooFormulaVar* nll,
           const char* varToscan,
           double scan_min,
           double scan_max,
           unsigned int npoints,
           double varGenVal=DUMMY_VALUE);

    /// Constructor without the points of the grid
    PLScan(const char* name,
           const char* title,
           RooFormulaVar* nll,
           const char* varToscan,
           double varGenVal=DUMMY_VALUE);

    /// Set the points to scan after construction
    void setPoints(std::vector<double> points);

    ///  Set the points to scan after construction
    void setPoints(double scan_min,
                   double scan_max,
                   unsigned int npoints);

    /// Perform the Likelihood scan
    PLScanResults* doScan(bool profile=true);

    /// Print the information about the object
    void print(const char* options);

  private:

    /// The grid points
    std::vector<double> m_points_grid;

    /// The NLL values at the points
    std::vector<double> m_NLL_values;

    /// The stored generated value
    double m_generated_value;

    /// The stored name of the scanned parameter
    TString m_scanned_parameter_name;

    /// The stored nll
    RooFormulaVar* m_nll;

    // For Cint
    //ClassDef(PLScan,1)
 };

#endif

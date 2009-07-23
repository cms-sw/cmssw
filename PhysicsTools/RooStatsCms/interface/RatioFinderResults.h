/// RatioFinderResults: The results of the SM production cross sections.
/**
\class RatioFinderResults
$Revision: 1.3 $
$Date: 2009/04/15 11:10:45 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

Collect the info out of a RatioFinder instance.
**/

#ifndef __RatioFinderResults__
#define __RatioFinderResults__

#include <map>

#include "RooAbsPdf.h"
#include "RooArgList.h"

#include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"
#include "PhysicsTools/RooStatsCms/interface/ConstrBlockArray.h"
#include "PhysicsTools/RooStatsCms/interface/RatioFinderPlot.h"


class RatioFinderResults : public StatisticalMethod {

  public:

    /// Constructor
    RatioFinderResults(const char* name,
                       const char* title,
                       int n_toys,
                       double CL_level,
                       double upper_ratio,
                       double lower_CL,
                       double lower_ratio,
                       double upper_CL,
                       double delta_ratios_min,
                       std::map<double,double> points);

    /// Default Ctor
    RatioFinderResults();

    /// Get the interpolated ratio
    double getInterpolatedRatio(const char* option);

    /// Get upper ratio
    double getUpperRatio(){return m_upper_ratio;};

    /// Get lower ratio
    double getLowerRatio(){return m_lower_ratio;};

    /// Get upper CL
    double getUpperCl(){return m_upper_CL;};

    /// Get lower CL
    double getLowerCl(){return m_lower_CL;}

    /// Get the requested CL
    double getCL(){return m_CL_level;};

    /// Get plot
    RatioFinderPlot* getPlot(const char* name="",const char* title="");

    /// Destructor
    ~RatioFinderResults();

    /// Print relevant information
    void print(const char* options="");

  private:

    /// The number of toys
    int m_n_toys;

    /// The requested CL
    double m_CL_level;

    /// The upper ratio
    double m_upper_ratio;

    /// The lower ratio
    double m_lower_ratio;

    /// The upper CL
    double m_upper_CL;

    /// The lower CL
    double m_lower_CL;

    /// The epsilon between the 2 ratios
    double m_delta_ratios_min;

    /// The map of the points for the TGraph
    std::map<double,double> m_points;



 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

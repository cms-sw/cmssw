// Author: Danilo.Piparo@cern.ch 20/01/2008

/// LimitResults: a LimitCalculator results collector

/**
\class LimitResults
$Revision: 1.1.1.1 $
$Date: 2009/04/15 08:40:01 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

The objects of this class store and access with lightweight methods the 
information calculated by LimitResults through a Lent calculation using 
MC toy experiments.
In some ways can be considered an extended and extensible implementation of the 
TConfidenceLevel class (http://root.cern.ch/root/html/TConfidenceLevel.html).
**/

#ifndef __LimitResults__
#define __LimitResults__

#include <vector>

#include "TLine.h"
#include "TH1F.h"
#include "TLatex.h"

#include "PhysicsTools/RooStatsCms/interface/StatisticalMethod.h"
#include "PhysicsTools/RooStatsCms/interface/LimitPlot.h"

class LimitResults : public StatisticalMethod {

  public:

    /// Constructor 
    LimitResults(const char* name,
                 const char* title,
                 std::vector<float>& m2lnq_sb_vals,
                 std::vector<float>& m2lnq_b_vals,
                 float m2lnq_data);

    /// Default Constructor 
    LimitResults();

    /// Destructor
    ~LimitResults();

    /// Get the CLsb value
    double getCLsb();

    /// Get the CLb value
    double getCLb();

    /// Get the CLs value
    double getCLs();

    /// Get the plot object pointer
    LimitPlot* getPlot(const char* name="",const char* title="", int n_bins=100);

    /// Get -2lnQ values for the sb model
    std::vector<float> getM2lnQValues_sb(){return m_m2lnQ_sb;}

    /// Get -2lnQ values for the b model
    std::vector<float> getM2lnQValues_b(){return m_m2lnQ_b;}

    /// Get -2lnQ value on Data
    double getM2lnQValue_data(){return m_m2lnQ_data;}

    /// Set -2lnQ value on Data
    void setM2lnQValue_data(double new_val){m_m2lnQ_data=new_val;}

    /// Add a second result to the present one
    void add(LimitResults* other);

    /// Print relevant info
    void print(const char* options="");

 private:

    /// Build CLb
    void m_build_CLb();

    /// The cached Clb Val
    double m_CLb;

    /// Build CLsb
    void m_build_CLsb();

    /// The cached ClSb Val
    double m_CLsb;

    /// -2lnQ values for sb model
    std::vector<float> m_m2lnQ_sb;

    /// -2lnQ values for b model
    std::vector<float> m_m2lnQ_b;

    /// -2lnQ calculated on data
    float m_m2lnQ_data;
/*
    /// Sorting flag
    bool m_vectors_sorted;

    /// Sorting criterium
    bool m_sorting_criterium(float a, float b){return a<b;};

    /// Calculate median of the vector
    double m_getMedian(std::vector<float>& vals);

    /// Calculate mean of the vector
    double m_getMean(std::vector<float>& vals);

    /// Calculate rms of the vector
    double m_getRMS(std::vector<float>& vals);
*/

};

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

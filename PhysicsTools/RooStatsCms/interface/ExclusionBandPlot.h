/// ExclusionBandPlot: plot a la tevatron for SM eclusion in function of mass

/**
\class ExclusionBandPlot
$Revision: 1.5 $
$Date: 2009/09/23 19:41:22 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

Yet another band plot, this time for sm exclusion, following the examples of 
the Tevatron Higgs WG.

**/

#ifndef __ExclusionBandPlot__
#define __ExclusionBandPlot__

#include <iostream>

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "StatisticalPlot.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/StatisticalPlot.h"
#endif

#include "TGraphErrors.h"
#include "TLine.h"
#include "TLegend.h"

class ExclusionBandPlot : public StatisticalPlot {

  public:

    /// Constructor
    ExclusionBandPlot(const char* name,
                      const char* title,
                      const int n_points,
                      double* x_vals,
                      double* y_vals,
                      double* y_up_bars1,
                      double* y_down_bars1,
                      double* y_up_bars2,
                      double* y_down_bars2);

    /// Set the title of the x axis
    void setXaxisTitle(const char* title);

    /// Set the title of the x axis
    void setYaxisTitle(const char* title);

    /// Set the title of the plot
    void setTitle(const char* title);

    /// Destructor
    ~ExclusionBandPlot();

    /// Draw on canvas
    void draw (const char* options="");

    /// Print the relevant information
    void print (const char* options="");

    /// All the objects are written to rootfile
    void dumpToFile (const char* RootFileName, const char* options);

  private:

    /// The line
    TGraph* m_y_line_graph;

    /// The band 1 sigma
    TGraph* m_y_band_graph_1sigma;

    /// The band 2 sigma
    TGraph* m_y_band_graph_2sigma;

    /// The line at 1
    TLine* m_one_line;

    /// The legend
    TLegend* m_legend;



//For Cint/**/
// #if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(ExclusionBandPlot,1)
// #endif
 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

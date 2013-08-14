/// LEPBandPlot: The plot for the CL bands "a la LEP"

/**
\class LEPBandPlot
$Revision: 1.5 $
$Date: 2009/09/23 19:41:22 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class allows to produce plots like the ones of the 
<a href="http://tevnphwg.fnal.gov/">Tevatron HWG </a> and originally introduced 
by <a href="http://lephiggs.web.cern.ch/LEPHIGGS/www/Welcome.html">LEP HWG</a>.
It is thought as a "enhanced" TGraph. The input to give to obtain the 
plot are:
 - The number of points
 - The values of the parameter on the x axis
 - The mean/median of the -2lnQ distribution of the sig+bkg hypothesis
 - The mean/median of the -2lnQ distribution of the bkg only hypothesis
 - The sigma (error bar size) of the -2lnQ distribution of the bkg only hypothesis
 - The experimental values of -2lnQ

\image html LEP_band_plot.png


**/

#ifndef __LEPBandPlot__
#define __LEPBandPlot__

#include <iostream>

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "StatisticalPlot.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/StatisticalPlot.h"
#endif

#include "TGraphErrors.h"
#include "TLine.h"
#include "TLegend.h"

class LEPBandPlot : public StatisticalPlot {

  public:

    /// Constructor
    LEPBandPlot(const char* name,
                const char* title,
                const int n_points,
                double* x_vals,
                double* sb_vals,
                double* b_vals,
                double* b_rms,
                double* exp_vals=0);
    /// Constructor
    LEPBandPlot(const char* name,
                const char* title,
                const int n_points,
                double* x_vals,
                double* sb_vals,
                double* b_vals,
                double* b_up_bars1,
                double* b_down_bars1,
                double* b_up_bars2,
                double* b_down_bars2,
                double* exp_vals=0);

    /// Set the title of the x axis
    void setXaxisTitle(const char* title);

    /// Set the title of the plot
    void setTitle(const char* title);

    /// Destructor
    ~LEPBandPlot();

    /// Draw on canvas
    void draw (const char* options="");

    /// Print the relevant information
    void print (const char* options="");

    /// All the objects are written to rootfile
    void dumpToFile (const char* RootFileName, const char* options);

  private:


    /// The data line
    TGraph* m_data_line_graph;

    /// The b line
    TGraph* m_b_line_graph;

    /// The b band 1 sigma
    TGraph* m_b_band_graph_1sigma;

    /// The b band 2 sigma
    TGraph* m_b_band_graph_2sigma;

    /// The sb line
    TGraph* m_sb_line_graph;

    /// The line at 0
    TLine* m_zero_line;

    /// The legend
    TLegend* m_legend;



//For Cint
// #if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(LEPBandPlot,1)
// #endif
 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

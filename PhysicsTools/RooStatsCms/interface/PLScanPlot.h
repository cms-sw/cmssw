/// PLScanPlot: The plot for the PLScan

/**
\class PLScanPlot
$Revision: 1.3 $
$Date: 2009/04/15 11:10:45 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe
**/

#ifndef __PLScanPlot__
#define __PLScanPlot__

#include <vector>
#include <iostream>

#include "PhysicsTools/RooStatsCms/interface/StatisticalPlot.h"

#include "TGraph.h"
#include "TLine.h"
#include "TText.h"

#include "TGraphErrors.h"

const int MAX_CL_LINES=10; // more than this...

class PLScanPlot : public StatisticalPlot {

  public:

    /// Constructor
    PLScanPlot(const char* name,
               const  char* title,
               const char* scanned_var,
               std::vector<double> x_points,
               std::vector<double> y_points,
               double var_at_min,
               int float_digits=2,
               bool verbosity=true);

    /// Destructor
    ~PLScanPlot();

    /// Add Cl line, CL in percentage
    int addCLline(double deltaNll,double CL, double limit);

    /// Add Cl line, CL in percentage
    int addCLline(double deltaNll, double CL, double Ulimit, double Llimit);

    /// Add FC graph, CL in percentage
    int addFCgraph(TGraphErrors* FC_graph, double CL);

    /// Draw on canvas
    void draw (const char* options="");

    /// Print the relevant information
    void print (const char* options="");

    /// All the objects are written to rootfile
    void dumpToFile (const char* RootFileName, const char* options);

  private:

    /// The number of the lines
    int m_cl_lines_num;

    /// The number of the line tags
    int m_cl_lines_tags_num;

    /// The number of FC graphs
    int m_fc_graphs_num;

    /// The number of float digits to display
    int m_float_digits;

    /// Value of the minimum
    double m_val_at_min;

    /// The tag for the minimum
    TText* m_minimum_tag;

    /// Name of the scanned var 
    char m_scanned_var[100];

    /// The graph of the scanned points
    TGraph* m_scan_graph;

    /// The graph of the scanned points
    TGraph* m_min_point_graph;

    /// Build the minimum tag
    void m_build_minimum_tag();

    /// The orizzontal lines for the confidence levels
    TLine* m_cl_lines[MAX_CL_LINES];

    /// The FC graphs for the confidence levels
    TGraphErrors* m_fc_graphs[MAX_CL_LINES];

    /// The orizzontal line to mark the 0
    TLine* m_zero_line;

    /// The tags for the orizzontal lines
    TText* m_cl_lines_tags[MAX_CL_LINES];

    double m_getDeltaNLLfromCL(double CL);



 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:33 2009

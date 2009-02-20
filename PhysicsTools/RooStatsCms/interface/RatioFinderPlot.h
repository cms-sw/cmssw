/// RatioFinderPlot: The results of the SM production cross sections.
/**
\class RatioFinderPlot
$Revision: 1.1 $
$Date: 2009/01/06 12:18:37 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

Collect the info out of a RatioFinder instance.
**/

#ifndef __RatioFinderPlot__
#define __RatioFinderPlot__

#include <map>

#include "TGraph.h"
#include "TLine.h"
#include "TAxis.h"

#include "PhysicsTools/RooStatsCms/interface/StatisticalPlot.h"


class RatioFinderPlot : public StatisticalPlot {

  public:

    /// Constructor
    RatioFinderPlot(const char* name,
                    const char* title,
                    double CL_level,
                    std::map<double,double> points);

    /// Draw on canvas
    void draw (const char* options="");

    /// Print the relevant information
    void print (const char* options="");

    /// All the objects are written to rootfile
    void dumpToFile (const char* RootFileName, const char* options);

    /// Set Title
    void setTitle(const char* title){m_graph->SetTitle(title);}

    /// Set Y axis yitle
    void setYaxisTitle(const char* title){m_graph->GetYaxis()->SetTitle(title);}

    /// Destructor
    ~RatioFinderPlot();

  private:

    /// The Graph
    TGraph* m_graph;

    /// The orizontal line for the CL level requested
    TLine* m_CL_line;

    // For Cint
    //ClassDef(RatioFinderPlot,1)
 };

#endif

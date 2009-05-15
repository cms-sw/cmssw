/// RatioFinderPlot: The results of the SM production cross sections.
/**
\class RatioFinderPlot
$Revision: 1.3 $
$Date: 2009/04/15 11:10:45 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

Collect the info out of a RatioFinder instance.
**/

#ifndef __RatioFinderPlot__
#define __RatioFinderPlot__

#include <map>

#include "TGraph.h"
#include "TLine.h"
#include "TAxis.h"

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "StatisticalPlot.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/StatisticalPlot.h"
#endif


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

//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(RatioFinderPlot,1)
#endif
 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

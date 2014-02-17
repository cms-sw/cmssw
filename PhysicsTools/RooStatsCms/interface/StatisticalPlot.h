/// StatisticalPlot: the base class for the statistical plots

/**
\class StatisticalPlot
$Revision: 1.4 $
$Date: 2009/05/15 09:55:43 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class provides the base for all the statistical Plots.
**/

#ifndef __StatisticalPlot__
#define __StatisticalPlot__


#include "TNamed.h"
#include "TCanvas.h"

#include "TFile.h"

class StatisticalPlot : public TNamed {

  public:

    /// Constructor
    StatisticalPlot(const char* name,const  char* title,bool verbosity=true);

    /// Destructor
    ~StatisticalPlot();

    /// Set the verbosity
    void setVerbosity(bool verbosity);

    /// get the verbosity
    bool is_verbose();

    /// Get the canvas
    TCanvas* getCanvas(){return m_canvas;}

    /// Set the canvas
    void setCanvas(TCanvas* new_canvas){m_canvas=new_canvas;}

    /// Write an image on disk
    void dumpToImage (const char* filename){m_canvas->Print(filename);}

    /// Draw on canvas
    virtual void draw (const char* options="") = 0;

    /// Print the relevant information
    virtual void print (const char* options="") = 0;

    /// All the objects are written to rootfile
    virtual void dumpToFile (const char* RootFileName, const char* options="") = 0;

  private:

    /// Verbosity flag
    bool m_verbose;

    /// Canvas
    TCanvas* m_canvas; 

//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(StatisticalPlot,1)
#endif
 };

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009

/// LimitPlot: The plot for the PLScan

/**
\class LimitPlot
$Revision: 1.1 $
$Date: 2009/01/06 12:18:37 $
\author D. Piparo (danilo.piparo<at>cern.ch), G. Schott - Universitaet Karlsruhe

This class provides the plots for the result of a study performed with the 
LimitCalculator class.
**/

#ifndef __LimitPlot__
#define __LimitPlot__

#include <vector>
#include <iostream>

#include "PhysicsTools/RooStatsCms/interface/Rsc.h"

#include "PhysicsTools/RooStatsCms/interface/StatisticalPlot.h"

#include "TH1F.h"
#include "TLine.h"
#include "TText.h"
#include "TLegend.h"


class LimitPlot : public StatisticalPlot {

  public:

    /// Constructor
    LimitPlot(const char* name,
              const char* title,
              std::vector<float> sb_values,
              std::vector<float> b_values,
              float m2lnQ_data,
              int n_bins,
              bool verbosity=true);

    /// Destructor
    ~LimitPlot();

    /// Draw on canvas
    void draw (const char* options="");

    /// Print the relevant information
    void print (const char* options="");

    /// All the objects are written to rootfile
    void dumpToFile (const char* RootFileName, const char* options);

    /// Get B histo mean
    double getBmean(){return m_b_histo->GetMean();};

    /// Get B histo RMS
    double getBrms(){return m_b_histo->GetRMS();};

    /// Get B histo
    TH1F* getBhisto(){return m_b_histo;}

    /// Get B histo center
    double getBCenter(double n_sigmas=1, bool display=false)
                              {return Rsc::getHistoCenter(m_b_histo,n_sigmas,display);};

    /// Get B histo integration extremes to obtain the requested area fraction
    double* getBIntExtremes(double frac)
                                   {return Rsc::getHistoPvals(m_b_histo,frac);};

    /// Get SB histo mean
    double getSBmean(){return m_sb_histo->GetMean();};

    /// Get SB histo center
    double getSBCenter(double n_sigmas=1, bool display=false)
                             {return Rsc::getHistoCenter(m_sb_histo,n_sigmas,display);};

    /// Get SB histo RMS
    double getSBrms(){return m_sb_histo->GetRMS();};

    /// Get SB histo integration extremes to obtain the requested area fraction
    double* getSBIntExtremes(double frac)
                                  {return Rsc::getHistoPvals(m_sb_histo,frac);};

    /// Get B histo
    TH1F* getSBhisto(){return m_sb_histo;}

  private:

    /// The sb Histo
    TH1F* m_sb_histo;

    /// The sb Histo shaded
    TH1F* m_sb_histo_shaded;

    /// The b Histo
    TH1F* m_b_histo;

    /// The b Histo shaded
    TH1F* m_b_histo_shaded;

    /// The line for the data -2lnQ
    TLine* m_data_m2lnQ_line;

    /// The legend of the plot
    TLegend* m_legend;

    // For Cint
    //ClassDef(LimitPlot,1) 
 };

#endif

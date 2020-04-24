#ifndef DrawPlot_h
#define DrawPlot_h



#include <vector>


#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TLegend.h"


class DrawPlot{
  public:
    DrawPlot(const unsigned int =0, const bool =true);
    ~DrawPlot();
    
    void setLegendEntry(const TString&, const TString&, const TString&);
    void setLegendCoordinate(const double, const double, const double, const double);
    void drawPlot(const TString&, const TString&, bool =true, bool =true);
    void drawTrackPlot(const TString&, const TString&, const bool =true, const bool =true);
    void drawEventPlot(const TString&, const TString&, const bool =true, const bool =true);
    
    void thesisMode(){thesisMode_ = true;}
    
  private:
    struct LegendEntries{
      LegendEntries(): legendEntry(""), legendEntryZeroApe(""), designLegendEntry(""){}
      TString legendEntry;
      TString legendEntryZeroApe;
      TString designLegendEntry;
    };
    
    void printHist(const TString&, const TString&, const bool, const bool);
    void scale(std::vector<TH1*>&, const double =1.)const;
    double maximumY(std::vector<TH1*>&)const;
    double minimumY(std::vector<TH1*>&)const;
    void setRangeUser(std::vector<TH1*>&, const double, const double)const;
    void setLineWidth(std::vector<TH1*>&, const unsigned int)const;
    void draw(std::vector<TH1*>&)const;
    void cleanup(std::vector<TH1*>&)const;
    
    LegendEntries adjustLegendEntry(const TString&, TH1*&, TH1*&, TH1*&);
    void adjustLegend(TLegend*&)const;
    
    
    const TString* outpath_;
    // File with distributions for result after iterations (final APE)
    TFile* file_;
    // File with distributions for result before iterations (APE=0)
    TFile* fileZeroApe_;
    // File with distributions for design geometry
    TFile* designFile_;
    // Only used when baseline should be drawn in residualWidth plot
    TTree* baselineTreeX_;
    TTree* baselineTreeY_;
    double* delta0_;
    
    // For setting legend in plots
    TString legendEntry_;
    TString legendEntryZeroApe_;
    TString designLegendEntry_;
    double legendXmin_;
    double legendYmin_;
    double legendXmax_;
    double legendYmax_;
    
    bool thesisMode_;
};





#endif




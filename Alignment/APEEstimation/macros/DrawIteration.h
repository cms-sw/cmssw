#ifndef DrawIteration_h
#define DrawIteration_h



#include <vector>
#include <map>

#include "TString.h"
#include "TFile.h"
#include "TGraph.h"



class DrawIteration{
  public:
    DrawIteration(unsigned int =0, const bool =false);
    ~DrawIteration();
    
    void yAxisFixed(const bool yAxis){yAxisFixed_ = yAxis;}
    
    void drawIteration(unsigned int =0, unsigned int =99999);
    void drawResult();
    
    void addSystematics();
    void addCmsText(const TString&);
    
    void addInputFile(const TString&, const TString&);
    void outputDirectory(const TString&);
    
  private:
    struct ExtremeValues{
      ExtremeValues(const double minApe, const double maxApe, const double maxAbsCorr):
      minimumApe(minApe), maximumApe(maxApe), maxAbsCorrection(maxAbsCorr){}
      const double minimumApe;
      const double maximumApe;
      const double maxAbsCorrection;
    };
    
    struct SectorValues{
      SectorValues(){}
      std::map<unsigned int, std::string*> m_sectorName;
      std::map<unsigned int, std::vector<double> > m_sectorValueX;
      std::map<unsigned int, std::vector<double> > m_sectorValueY;
    };
    
    SectorValues getSectorValues(TFile*);
    ExtremeValues getGraphs(const std::string, unsigned int, unsigned int);
    void drawCorrections(const std::string&, const ExtremeValues&, const std::string&);
    void printFinalValues(unsigned int, unsigned int);
    void clear();
    
    std::vector<std::vector<std::string> > arrangeHists();
    std::vector<std::string> pixelHist();
    std::vector<std::string> barrelHist();
    std::vector<std::string> tibHist();
    std::vector<std::string> tobHist();
    std::vector<std::string> endcapHist();
    std::vector<std::string> tidHist();
    std::vector<std::string> tecHist();
    TString associateLabel(const std::string&);
    unsigned int sectorNumber(const std::string&);
    void drawFinals(const std::string&);
    bool createResultHist(TH1*&, const std::vector<std::string>&, const std::string&, SectorValues&, unsigned int);

    const TString* outpath_;
    TFile* file_;
    const bool overlayMode_;
    bool yAxisFixed_;
    
    SectorValues sectorValues_;
    
    std::vector<TGraph*> v_graphApeX_;
    std::vector<TGraph*> v_graphCorrectionX_;
    std::vector<TGraph*> v_graphApeY_;
    std::vector<TGraph*> v_graphCorrectionY_;
    
    std::vector<std::vector<std::string> > v_resultHist_;
    bool systematics_;
    TString cmsText_;
    
    struct Input{
      Input(TString name, TString legend): fileName(name), legendEntry(legend), file(0){}
      
      TString fileName;
      TString legendEntry;
      TFile* file;
      
      SectorValues sectorValues;
    };
    
    std::vector<Input*> v_input_;
};





#endif





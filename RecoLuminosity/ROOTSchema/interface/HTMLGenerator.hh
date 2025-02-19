#ifndef _WBMHTMLGENERATOR_H_
#define _WBMHTMLGENERATOR_H_

#include "RecoLuminosity/ROOTSchema/interface/FileToolKit.h"
#include "RecoLuminosity/TCPReceiver/interface/TimeStamp.h"

// ROOT Schema Headers
#include <vector>
#include <string>
#include <sys/stat.h>

class TH1F;
class TCanvas;
class TH2F;

namespace HCAL_HLX{
  
  class ROOTFileReader;
  struct LUMI_SECTION;

  class HTMLGenerator: private FileToolKit, private TimeStamp{
  public:
    HTMLGenerator();
    
    ~HTMLGenerator();
    
    void CreateWebPage( const std::string &fileName, const unsigned int iEntry = 0);

    void SetInputDir(  const std::string &inputDir);
    void SetOutputDir( const std::string &outDir){ outputDir_ = outDir; }
    void SetEtSumOnly( const bool bEtSumOnly );
    void SetHistoBins( unsigned int NBins, double XMin, double XMax);
    void SetFileType(  const std::string &fileType );

  private:
    
    ROOTFileReader *RFReader_; 
    bool bEtSumOnly_;

    std::string GetRunDir();
    std::string GetLSDir();
    std::string GetHLXDir(const unsigned short int &wedgeNum);
    std::string GetHLXPicDir(const unsigned short int &wedgeNum);
    
    void GenerateIndexPage();
    
    void GenerateRunPage();
    void GenerateSectionPage();
    
    void GenerateAveragePlots();
    void GenerateAveragePage();
    
    void GenerateComparePlots();
    void GenerateComparePage();
    
    void GenerateHLXPage(const unsigned short int &wedgeNum);
    void GenerateHLXPlots(const unsigned short int &wedgeNum);
    
    //       void GenerateEtSumPage();
    
    void GenerateHistoGroupPage(const std::string &histoType);
    void GenerateLumiPage();
    
    std::string fileName_;
    std::string outputFile_;
    std::string outputDir_;
    std::string inputDir_;
    std::string plotExt_;
    std::string baseURL;
    
    mode_t writeMode_;
    
    std::vector< std::string > HistoNames;
    
    unsigned int  previousRunNumber;
    
    unsigned int NBins_;
    double XMin_;
    double XMax_;
    unsigned int BinWidth_;
    
    int iEta_[36];
    int iPhi_[36];
    
    HCAL_HLX::LUMI_SECTION *lumiSection_;
    
    std::string HLXToHFMap_[36];
    
    TH1F* HLXHistos_[8];
    TCanvas* c1_;
    
    TH2F* EtSummary_;
    TH2F* OccSummary_;
    TH2F* MaxEtSummary_;
    TH2F* MaxLHCSummary_;
    
    TH1F* ETLumiHisto_;
    TH1F* OccLumiSet1Histo_;
    TH1F* OccLumiSet2Histo_;
    
  };  
}

#endif

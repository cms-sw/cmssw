#ifndef _WBMHTMLGENERATOR_H_
#define _WBMHTMLGENERATOR_H_

// STL Headers
#include <string>

// ROOT Schema Headers
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileReader.h"
#include "RecoLuminosity/TCPReceiver/interface/TimeStamp.h"

namespace HCAL_HLX{

  class HTMLGenerator: public ROOTFileReader, private TimeStamp{
  public:
    HTMLGenerator();
    
    ~HTMLGenerator();
    
    void CreateWebPage();
    
  private:
    
    std::string GetRunDir();
    int MakeRunDir();
    
    std::string GetLSDir();
    int MakeLSDir();
    
    std::string GetWedgeDir(const unsigned short int &wedgeNum);
    int MakeWedgeDir(const unsigned short int &wedgeNum);
    
    std::string GetWedgePicDir(const unsigned short int &wedgeNum);
    int MakeWedgePicDir(const unsigned short int &wedgeNum);
    
    void GenerateIndexPage();
    
    void GenerateRunPage();
    void GenerateSectionPage();
    
    void GenerateAveragePlots();
    void GenerateAveragePage();
    
    void GenerateComparePlots();
    void GenerateComparePage();
    
    void GenerateWedgePage(const unsigned short int &wedgeNum);
    void GenerateWedgePlots(const unsigned short int &wedgeNum);
    
    std::string fileName_;
    std::string outputFile_;
    std::string outputDir_;
    std::string plotExt_;
    std::string baseURL;
    
    unsigned int runNum;
    unsigned int LSNum;
    
    mode_t writeMode;
  };
  
}

#endif

#ifndef __ROOTFILEMERGER_H__
#define __ROOTFILEMERGER_H__

// STL Headers
#include <string>


namespace HCAL_HLX{

  class ROOTFileReader;
  class ROOTFileWriter;
  struct LUMI_SECTION;

  class ROOTFileMerger{
  public:
    ROOTFileMerger();
    ~ROOTFileMerger();
    
    void SetInputDir( const std::string &inputDir);
    void SetOutputDir( const std::string &outputDir);
    void SetEtSumOnly( bool bEtSumOnly);

    std::string GetInputFileName();
    std::string GetOutputFileName();
    
    void Merge(const unsigned int runNumber, bool bCMSLive);
    
  private:

    void Init();
    void CleanUp();

    unsigned int minSectionNumber_;
    
    ROOTFileWriter *RFWriter_;
    ROOTFileReader *RFReader_;

    LUMI_SECTION *lumiSection_;

  };
}

#endif

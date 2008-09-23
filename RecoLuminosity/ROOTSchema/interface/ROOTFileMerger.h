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
    
    void SetInputDir(  const std::string &inputDir);
    void SetOutputDir( const std::string &outputDir);

    void SetEtSumOnly( const bool bEtSumOnly );
    void SetFileType(  const std::string &fileType);
    void SetDate(      const std::string & date);

    std::string GetInputFileName();
    std::string GetOutputFileName();
    
    bool Merge(const unsigned int runNumber, const unsigned int minSecNum);
    
  private:

    unsigned int minSectionNumber_;
    
    ROOTFileWriter *RFWriter_;
    ROOTFileReader *RFReader_;

    LUMI_SECTION *lumiSection_;

  };
}

#endif

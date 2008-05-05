#ifndef __ROOTFILEMERGER_H__
#define __ROOTFILEMERGER_H__

// STL Headers
#include <string>

#include "RecoLuminosity/TCPReceiver/interface/TimeStamp.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileReader.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileBase.h"

namespace HCAL_HLX{
  
  class ROOTFileMerger: public ROOTFileReader, public ROOTFileBase{
  public:
    ROOTFileMerger();
    ~ROOTFileMerger();
    
    void Merge(const unsigned int runNumber, bool bCMSLive);
    
  private:
    void MergeMethodOne(const unsigned int runNumber, bool bCMSLive);
    void MergeMethodTwo(const unsigned int runNumber, bool bCMSLive);
    void MergeMethodThree(const unsigned int runNumber, bool bCMSLive);
    
    std::string CreateInputFileName(const unsigned int runNumber);
    
  };
}

#endif

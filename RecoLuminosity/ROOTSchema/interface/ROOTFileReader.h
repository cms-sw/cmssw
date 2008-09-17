#ifndef _ROOTFILEREADER_H_
#define _ROOTFILEREADER_H_


// STL Headers
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

// Lumi Headers
#include "RecoLuminosity/TCPReceiver/interface/ICTypeDefs.hh"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileBase.h"

class TChain;
class TBranch;

namespace HCAL_HLX{

  class ROOTFileReader: public ROOTFileBase{
  public:
    ROOTFileReader();
    ~ROOTFileReader();

    // For automatic replacement of files.
    bool SetRunNumber( const unsigned int runNumber, const std::string &month = "");
    // For manual replacment of files.
    int ReplaceFile(const std::string &fileName);
    int ReplaceFile(const std::vector< std::string > &fileNames);

    int GetThreshold(HCAL_HLX::LUMI_THRESHOLD&   threshold);
    int GetHFRingSet(HCAL_HLX::LUMI_HF_RING_SET& ringSet);
    int GetL1Trigger(HCAL_HLX::LEVEL1_TRIGGER&   l1trigger);
    int GetHLT(HCAL_HLX::HLT& hlt);
    int GetTriggerDeadtime(HCAL_HLX::TRIGGER_DEADTIME& TD);

    int GetEntry(int entry, HCAL_HLX::LUMI_SECTION& section);
    int GetNumEntries();
    int GetFirstSectionNumber(){ return firstSecNum_; }
    
  private:    

    int CreateFileNameList( const std::string &runDir);
    void CreateTree();

    unsigned int numEntries_;
    unsigned int firstSecNum_;

    TChain* mChain_;

    // Branches
    // LUMI_SECTION
    TBranch *b_Header;
    TBranch *b_Summary;
    TBranch *b_Detail;
    
    TBranch *b_ETSum[HCAL_HLX_MAX_HLXS];
    TBranch *b_Occupancy[HCAL_HLX_MAX_HLXS];
    TBranch *b_LHC[HCAL_HLX_MAX_HLXS];

    // OTHER
    TBranch *b_Threshold;
    TBranch *b_L1Trigger;
    TBranch *b_HLT;
    TBranch *b_TriggerDeadtime;
    TBranch *b_RingSet;


  };
}

#endif

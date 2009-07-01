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

    // For manual replacment of files.
    int SetFileName(const std::string &fileName);
    int CreateFileNameList(); // Call after SetDir.

    int GetEntry(int entry);

    int GetLumiSection( HCAL_HLX::LUMI_SECTION& section);
    int GetThreshold(HCAL_HLX::LUMI_THRESHOLD&   threshold);
    int GetHFRingSet(HCAL_HLX::LUMI_HF_RING_SET& ringSet);
    int GetL1Trigger(HCAL_HLX::LEVEL1_TRIGGER&   l1trigger);
    int GetHLT(HCAL_HLX::HLT& hlt);
    int GetTriggerDeadtime(HCAL_HLX::TRIGGER_DEADTIME& TD);

    unsigned int GetEntries();
    
  private:    

    int ReplaceFile(const std::vector< std::string > &fileNames);
    void CreateTree();

    TChain *mChain_;

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

#ifndef _ROOTFILEREADER_H_
#define _ROOTFILEREADER_H_


// STL Headers
#include <string>
#include <sstream>
#include <iomanip>

// ROOT Headers
#include <TChain.h>

// Lumi Headers
#include "RecoLuminosity/TCPReceiver/interface/ICTypeDefs.hh"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

namespace HCAL_HLX{

  class ROOTFileReader{
  public:
    ROOTFileReader();
    ~ROOTFileReader();
    
    int ReplaceFile(const std::string &fileName);
        
    unsigned int GetRunNumber(){ return runNumber_;}
    unsigned int GetSectionNumber(){ return sectionNumber_;}

    int GetLumiSection(HCAL_HLX::LUMI_SECTION& section);
    
    int GetThreshold(HCAL_HLX::LUMI_THRESHOLD&   threshold);
    int GetHFRingSet(HCAL_HLX::LUMI_HF_RING_SET& ringSet);
    int GetL1Trigger(HCAL_HLX::LEVEL1_TRIGGER&   l1trigger);
    int GetHLT(HCAL_HLX::HLT& hlt);
    int GetTriggerDeadtime(HCAL_HLX::TRIGGER_DEADTIME& TD);
    
    int GetEntry(int entry);
    int GetNumEntries();

  private:
    
    unsigned int runNumber_;
    unsigned int sectionNumber_;
    
    std::string mFileName_;
    TChain* mChain_;

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

    HCAL_HLX::LUMI_SECTION* lumiSection_;

    // LUMI_SECTION
    HCAL_HLX::LUMI_SECTION_HEADER* Header_;
    HCAL_HLX::LUMI_SUMMARY*        Summary_;
    HCAL_HLX::LUMI_DETAIL*         Detail_;

    HCAL_HLX::ET_SUM_SECTION      *EtSumPtr[HCAL_HLX_MAX_HLXS];
    HCAL_HLX::OCCUPANCY_SECTION   *OccupancyPtr[HCAL_HLX_MAX_HLXS];
    HCAL_HLX::LHC_SECTION         *LHCPtr[HCAL_HLX_MAX_HLXS];

    // Other
    HCAL_HLX::LUMI_THRESHOLD*   Threshold_;
    HCAL_HLX::LEVEL1_TRIGGER*   L1Trigger_;
    HCAL_HLX::HLT*              HLT_;
    HCAL_HLX::TRIGGER_DEADTIME* TriggerDeadtime_;
    HCAL_HLX::LUMI_HF_RING_SET* RingSet_;

  };
}

#endif


/*
  LumiNibble structure definitions
*/

#ifndef LUMISTRUCTURES_HH
#define LUMISTRUCTURES_HH

// The string and stream definitions
#include <iostream>
#include <string>

// Type definitions used by the HAL, etc...
#ifndef ICTypeDefs_HH  // CMSSW compatible
#include "ICTypeDefs.hh"
#endif

#define HCAL_HLX_MAX_BUNCHES 4096
#define HCAL_HLX_MAX_HLXS 36

//#define HCAL_HLX_NUM_BUNCHES 3564
//#define HCAL_HLX_NUM_HLXS 36

// Changes
// Namespace for the HCAL HLX
namespace HCAL_HLX
{
  // We will be using the IC core utility library
  using namespace ICCoreUtils;

  struct LUMI_SUMMARY {
    float DeadtimeNormalization; 
    float LHCNormalization; // received from LHC 

    float InstantLumi;
    float InstantLumiErr;
    i16 InstantLumiQlty;

    float InstantETLumi;
    float InstantETLumiErr;
    i16 InstantETLumiQlty;
    float ETNormalization;  // Calculated

    float InstantOccLumi[2];
    float InstantOccLumiErr[2];
    i16 InstantOccLumiQlty[2];
    float OccNormalization[2];

    float lumiNoise[2];
  };

  struct LUMI_DETAIL {
    float LHCLumi[HCAL_HLX_MAX_BUNCHES]; // Sum of LHC.data over all HLX's

    float ETLumi[HCAL_HLX_MAX_BUNCHES];
    float ETLumiErr[HCAL_HLX_MAX_BUNCHES];
    i16 ETLumiQlty[HCAL_HLX_MAX_BUNCHES];
    float ETBXNormalization[HCAL_HLX_MAX_BUNCHES];

    float OccLumi[2][HCAL_HLX_MAX_BUNCHES];
    float OccLumiErr[2][HCAL_HLX_MAX_BUNCHES];
    i16 OccLumiQlty[2][HCAL_HLX_MAX_BUNCHES];
    float OccBXNormalization[2][HCAL_HLX_MAX_BUNCHES];
  };

  struct LUMI_THRESHOLD {
    float OccThreshold1Set1;  // Occupancy Threshold
    float OccThreshold2Set1;
    float OccThreshold1Set2;
    float OccThreshold2Set2;
    float ETSum;
  };

  struct LUMI_HF_RING_SET{
    std::string Set1Rings;
    std::string Set2Rings;
    std::string EtSumRings;
  };

  struct LEVEL1_TRIGGER {
    std::string GTLumiInfoFormat;
    uint32_t  GTAlgoCounts[128];
    uint32_t GTAlgoPrescaling[128];
    uint32_t GTTechCounts[64];
    uint32_t GTTechPrescaling[64];
    uint32_t GTPartition0TriggerCounts[10];
    uint32_t GTPartition0DeadTime[10 ];
  };

  struct HLT{
//     PATHNAME  STRING  //This is the name of trigger path
//     L1PASS? LONG //Number of times the path was entered
//     PSPASS - LONG //Number after prescaling
//     PACCEPT LONG //Number of accepts by the trigger path
//     PEXCEPT LONG //Number of exceptional event encountered
//     PREJECT LONG //Number of rejected events
//     PRESCALERMODULE STRING //Name of the prescale module in the path
//     PSINDEX LONG //Index into the set of pre defined prescales

    std::string PathName;
    uint32_t L1Pass;
    uint32_t PSPass;
    uint32_t PAccept;
    uint32_t PExcept;
    uint32_t PReject;
    std::string PrescalerModule;
    uint32_t PSIndex;
    
  };

  struct TRIGGER_DEADTIME {
    int TriggerDeadtime;
  };

  struct LUMI_RAW_HEADER { // Used in NibbleCollector 
    u16 marker;
    u8  hlxID;
    u8  packetID;
    u32 startOrbit;
    u16 numOrbits;
    u16 startBunch; // Starting bunch in this packet
    u16 numBunches; // Total number of bunches in histogram
    u8  histogramSet;
    u8  histogramSequence;
    u16 allA;
    u16 allF;
  };

  struct LUMI_NIBBLE_HEADER {
    u32  startOrbit;
    u16  numOrbits;
    u16  numBunches; // Number of bunches histogrammed
    bool bCMSLive;
    bool bOC0;
  };
  
  struct ET_SUM_NIBBLE {
    LUMI_NIBBLE_HEADER hdr;
    u32 data[HCAL_HLX_MAX_BUNCHES];
  };
  
  struct OCCUPANCY_NIBBLE {
    LUMI_NIBBLE_HEADER hdr;
    u16 data[6][HCAL_HLX_MAX_BUNCHES];
  };

  struct LHC_NIBBLE {
    LUMI_NIBBLE_HEADER hdr;
    u16 data[HCAL_HLX_MAX_BUNCHES];
  };

  struct LUMI_SECTION_HEADER {
    u32 timestamp;
    u32 timestamp_micros;
    u32  runNumber;   // Run number
    u32  sectionNumber; // Section number
    u32  startOrbit;  // Start orbit of lumi section
    u32  numOrbits;   // Total number of orbits recorded in lumi section
    u16  numBunches;  // Total number of bunches (from start of orbit)
    u16  numHLXs;     // Number of HLXs in lumi section
    bool bCMSLive;    // Is CMS taking data?
    bool bOC0;        // Was section initialised by an OC0?

    uint16_t LS_Quality; // 0-999
    std::string RunSequenceName;  // Cosmic, Pedestal, ....
    uint32_t RunSequenceNumber; // Number that identifies version of RunSequenceName;
    std::string HLT_Config_KEY
    std::string CMSSW_Tag;
    std::string TriDAS_Tag;
  };

  struct LUMI_SECTION_SUB_HEADER {
    u32  numNibbles;  // Number of nibbles in this histogram
    bool bIsComplete; // Is this histogram complete (i.e. no missing nibbles)
  };

  struct ET_SUM_SECTION {
    LUMI_SECTION_SUB_HEADER hdr;
    float data[HCAL_HLX_MAX_BUNCHES];
  };

  struct OCCUPANCY_SECTION {
    LUMI_SECTION_SUB_HEADER hdr;
    u32 data[6][HCAL_HLX_MAX_BUNCHES];
  };

  struct LHC_SECTION {
    LUMI_SECTION_SUB_HEADER hdr;
    u32 data[HCAL_HLX_MAX_BUNCHES];
  };

  struct LUMI_SECTION {
    LUMI_SECTION_HEADER hdr;
    LUMI_SUMMARY lumiSummary;
    LUMI_DETAIL  lumiDetail;

    ET_SUM_SECTION etSum[HCAL_HLX_MAX_HLXS];
    OCCUPANCY_SECTION occupancy[HCAL_HLX_MAX_HLXS];
    LHC_SECTION lhc[HCAL_HLX_MAX_HLXS];
  };

}//~namespace HCAL_HLX

#endif //~LUMISTRUCTURES_HH


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
    float LHCNormalization; // recieved from LHC 

    float InstantLumi;
    float InstantLumiErr;
    u8 InstantLumiQlty;

    float InstantETLumi;
    float InstantETLumiErr;
    u8 InstantETLumiQlty;
    float ETNormalization;  // Calculated

    float InstantOccLumi[2];
    float InstantOccLumiErr[2];
    u8 InstantOccLumiQlty[2];
    float OccNormalization[2];

    float lumiNoise[2];
  };

  struct LUMI_DETAIL {
    float LHCLumi[HCAL_HLX_MAX_BUNCHES]; // Sum of LHC.data over all HLX's

    float ETLumi[HCAL_HLX_MAX_BUNCHES];
    float ETLumiErr[HCAL_HLX_MAX_BUNCHES];
    u8 ETLumiQlty[HCAL_HLX_MAX_BUNCHES];
    float ETBXNormalization[HCAL_HLX_MAX_BUNCHES];

    float OccLumi[2][HCAL_HLX_MAX_BUNCHES];
    float OccLumiErr[2][HCAL_HLX_MAX_BUNCHES];
    u8 OccLumiQlty[2][HCAL_HLX_MAX_BUNCHES];
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
    int L1lineNumber;
    int L1Scaler;
    int L1RateCounter;
  };

  struct HLT{
    int TriggerPath;
    int InputCount;
    int AcceptCount;
    int PrescaleFactor;
  };

  struct TRIGGER_DEADTIME {
    int TriggerDeadtime;
  };

  /*
  struct LUMI_SECTION_HST {
    bool IsDataTaking;
    int BeginOrbitNumber;
    int EndOrbitNumber;
    int RunNumber;
    int LumiSectionNumber;
    int FillNumber;
    int SecStopTime;
    int SecStartTime;
  };
  */

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
  };
  
  struct ET_SUM_NIBBLE {
    LUMI_NIBBLE_HEADER hdr;
    bool bIsComplete;
    u32 data[HCAL_HLX_MAX_BUNCHES];
  };
  
  struct OCCUPANCY_NIBBLE {
    LUMI_NIBBLE_HEADER hdr;
    bool bIsComplete[6];
    u16 data[6][HCAL_HLX_MAX_BUNCHES];
  };

  struct LHC_NIBBLE {
    LUMI_NIBBLE_HEADER hdr;
    bool bIsComplete;
    u16 data[HCAL_HLX_MAX_BUNCHES];
  };

  struct LUMI_SECTION_HEADER {
    u32  runNumber;   // Run number
    u32  sectionNumber; // Section number
    u32  startOrbit;  // Start orbit of lumi section
    u32  numOrbits;   // Total number of orbits recorded in lumi section
    u16  numBunches;  // Total number of bunches (from start of orbit)
    u16  numHLXs;     // Number of HLXs in lumi section
    bool bCMSLive;    // Is CMS taking data?
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

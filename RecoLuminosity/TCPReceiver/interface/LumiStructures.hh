/*
  LumiNibble structure definitions
*/

#ifndef LUMISTRUCTURES_HH
#define LUMISTRUCTURES_HH

// The string and stream definitions
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#define HCAL_HLX_MAX_BUNCHES 4096
#define HCAL_HLX_MAX_HLXS 36

//#define HCAL_HLX_NUM_BUNCHES 3564
//#define HCAL_HLX_NUM_HLXS 36

// Changes
// Namespace for the HCAL HLX
namespace HCAL_HLX {

  struct LUMI_SUMMARY {
    float DeadtimeNormalization;
    float LHCNormalization;  // received from LHC

    float InstantLumi;
    float InstantLumiErr;
    int16_t InstantLumiQlty;

    float InstantETLumi;
    float InstantETLumiErr;
    int16_t InstantETLumiQlty;
    float ETNormalization;  // Calculated
    float InstantOccLumi[2];
    float InstantOccLumiErr[2];
    int16_t InstantOccLumiQlty[2];
    float OccNormalization[2];

    float lumiNoise[2];
  };

  struct LUMI_DETAIL {
    float LHCLumi[HCAL_HLX_MAX_BUNCHES];  // Sum of LHC.data over all HLX's

    float ETLumi[HCAL_HLX_MAX_BUNCHES];
    float ETLumiErr[HCAL_HLX_MAX_BUNCHES];
    int16_t ETLumiQlty[HCAL_HLX_MAX_BUNCHES];
    float ETBXNormalization[HCAL_HLX_MAX_BUNCHES];

    float OccLumi[2][HCAL_HLX_MAX_BUNCHES];
    float OccLumiErr[2][HCAL_HLX_MAX_BUNCHES];
    int16_t OccLumiQlty[2][HCAL_HLX_MAX_BUNCHES];
    float OccBXNormalization[2][HCAL_HLX_MAX_BUNCHES];
  };

  struct LEVEL1_TRIGGER {
    std::string GTLumiInfoFormat;
    uint32_t GTAlgoCounts[128];
    uint32_t GTAlgoPrescaling[128];
    uint32_t GTTechCounts[64];
    uint32_t GTTechPrescaling[64];
    uint32_t GTPartition0TriggerCounts[10];
    uint32_t GTPartition0DeadTime[10];
  };

  struct HLTPath {
    std::string PathName;  //This is the name of trigger path
    uint32_t L1Pass;       //Number of times the path was entered
    uint32_t PSPass;       //Number after prescaling
    uint32_t PAccept;      //Number of accepts by the trigger path
    uint32_t PExcept;      //Number of exceptional event encountered
    uint32_t PReject;
    std::string PrescalerModule;  //Name of the prescale module in the path
    uint32_t PSIndex;             //Index into the set of pre defined prescales
  };

  struct HLT {
    std::string HLT_Config_KEY;
    std::vector<HLTPath> HLTPaths;
  };

  struct LUMI_RAW_HEADER {  // Used in NibbleCollector
    uint16_t marker;
    uint8_t hlxID;
    uint8_t packetID;
    uint32_t startOrbit;
    uint16_t numOrbits;
    uint16_t startBunch;  // Starting bunch in this packet
    uint16_t numBunches;  // Total number of bunches in histogram
    uint8_t histogramSet;
    uint8_t histogramSequence;
    uint16_t allA;
    uint16_t allF;
  };

  struct LUMI_NIBBLE_HEADER {
    uint32_t startOrbit;
    uint16_t numOrbits;
    uint16_t numBunches;  // Number of bunches histogrammed
    bool bCMSLive;
    bool bOC0;
  };

  struct ET_SUM_NIBBLE {
    LUMI_NIBBLE_HEADER hdr;
    uint32_t data[HCAL_HLX_MAX_BUNCHES];
  };

  struct OCCUPANCY_NIBBLE {
    LUMI_NIBBLE_HEADER hdr;
    uint16_t data[6][HCAL_HLX_MAX_BUNCHES];
  };

  struct LHC_NIBBLE {
    LUMI_NIBBLE_HEADER hdr;
    uint16_t data[HCAL_HLX_MAX_BUNCHES];
  };

  struct RUN_SUMMARY {
    uint16_t LS_Quality;         // 0-999
    char RunSequenceName[64];    // Cosmic, Pedestal, ....
    uint32_t RunSequenceNumber;  // Number that identifies version of RunSequenceName;
  };

  struct RCMS_CONFIG {
    std::string CMSSW_Tag;
    std::string TriDAS_Tag;

    bool UseConfigDB;
    std::string CfgDBTag;
    uint8_t FirmwareVersion;

    uint8_t OCCMaskTop;
    uint8_t OCCMaskBottom;

    uint8_t LHCMaskLowBottom;
    uint8_t LHCMaskLowTop;

    uint8_t LHCMaskHighBottom;
    uint8_t LHCMaskHighTop;

    uint16_t SumETMaskLowBottom;
    uint16_t SumETMaskLowTop;
    uint16_t SumETMaskHighBottom;
    uint16_t SumETMaskHighTop;

    uint16_t OccThresholdLowBottom;
    uint16_t OccThresholdLowTop;
    uint16_t OccThresholdHighBottom;
    uint16_t OccThresholdHighTop;

    uint16_t LHCThresholdBottom;
    uint16_t LHCThresholdTop;
    uint16_t ETSumCutoffBottom;
    uint16_t ETSumCutoffTop;
  };

  struct LUMI_SECTION_HEADER {
    uint32_t timestamp;
    uint32_t timestamp_micros;
    uint32_t runNumber;      // Run number
    uint32_t sectionNumber;  // Section number
    uint32_t startOrbit;     // Start orbit of lumi section
    uint32_t numOrbits;      // Total number of orbits recorded in lumi section
    uint16_t numBunches;     // Total number of bunches (from start of orbit)
    uint16_t numHLXs;        // Number of HLXs in lumi section
    bool bCMSLive;           // Is CMS taking data?
    bool bOC0;               // Was section initialised by an OC0?
  };

  struct LUMI_SECTION_SUB_HEADER {
    uint32_t numNibbles;  // Number of nibbles in this histogram
    bool bIsComplete;     // Is this histogram complete (i.e. no missing nibbles)
  };

  struct ET_SUM_SECTION {
    LUMI_SECTION_SUB_HEADER hdr;
    float data[HCAL_HLX_MAX_BUNCHES];
  };

  struct OCCUPANCY_SECTION {
    LUMI_SECTION_SUB_HEADER hdr;
    uint32_t data[6][HCAL_HLX_MAX_BUNCHES];
  };

  struct LHC_SECTION {
    LUMI_SECTION_SUB_HEADER hdr;
    uint32_t data[HCAL_HLX_MAX_BUNCHES];
  };

  struct LUMI_SECTION {
    LUMI_SECTION_HEADER hdr;
    LUMI_SUMMARY lumiSummary;
    LUMI_DETAIL lumiDetail;

    ET_SUM_SECTION etSum[HCAL_HLX_MAX_HLXS];
    OCCUPANCY_SECTION occupancy[HCAL_HLX_MAX_HLXS];
    LHC_SECTION lhc[HCAL_HLX_MAX_HLXS];
  };

}  // namespace HCAL_HLX

#endif  //~LUMISTRUCTURES_HH

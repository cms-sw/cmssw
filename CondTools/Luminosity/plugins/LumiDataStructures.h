#ifndef CondTools_Luminosity_LUMIDATASTRUCTURES_H
#define CondTools_Luminosity_LUMIDATASTRUCTURES_H
//Note: this header file corresponds to svn.cern.ch/reps/Luminosity/HLXReadOut/HLXCoreLibs/include/LumiStructures.hh - Revision 68 LumiDAQ-release-2.0
/*
  LumiNibble structure definitions
*/

// The string and stream definitions
// Type definitions used by the HAL, etc...

#define HCAL_HLX_MAX_BUNCHES 4096
#define HCAL_HLX_MAX_HLXS 36

//#define HCAL_HLX_NUM_BUNCHES 3564
//#define HCAL_HLX_NUM_HLXS 36

// Changes
// Namespace for the HCAL HLX
#include <stdint.h>

namespace HCAL_HLX{

  struct RUN_SUMMARY {
    char runSequenceName[128];
    char HLTKeyDescriptor[128];
    uint32_t timestamp;
    uint32_t timestamp_micros;
    uint32_t startOrbitNumber;
    uint32_t endOrbitnumber;
    uint32_t runNumber;
    uint32_t fillNumber;
    uint32_t numberCMSLumiSections;  // number of lumi sections from the trigger
    uint32_t numberLumiDAQLumiSections;
  };

  struct RUN_QUALITY {

    int HLX;
    int HFLumi;
    int ECAL;
    int HCAL;
    int Tracker;
    int RPC;
    int DT;
    int CSC;
  };

  struct VDM_SCAN_DATA {

    int MessageQuality;     
    bool VdmMode; //True when a scan at one of the IPs is imminent, false otherwise
    int IP;
    bool RecordDataFlag; // True while data for one of the scan points at one of the IPs is being taken, false otherwise   
    double BeamSeparation; //separation in sigma for the scan point  
    bool IsXaxis; //true if scanning xaxis, otherwise yaxis is being scanned 
    uint32_t timestamp;
    uint32_t timestamp_micros;
  };

  struct BRAN_DATA {

    int MessageQuality;     
    double MeanCrossingAngle;
    int AcqMode; 
    double MeanLuminosity;
    uint32_t timestamp;
    uint32_t timestamp_micros;
  };

  struct BRAN_BX_DATA {

    int MessageQuality;
    double bunchByBunchLuminosity[3564];
    int AcqMode;
    uint32_t timestamp;
    uint32_t timestamp_micros;
  };

  struct LHC_FILL_DATA {

    int MessageQuality;
    int FillNumber;
    uint32_t timestamp;
    uint32_t timestamp_micros;
  };

  struct RCMS_CONFIG {
    
    uint32_t runNumber;

    bool UseConfigDB;
    char CfgDBTag[32];

    char CMSSW_Tag[32];
    char TriDAS_Tag[32];

    uint16_t FirmwareVersion;
    
    uint16_t OCCMaskTop;
    uint16_t OCCMaskBottom;

    uint16_t LHCMaskLowBottom;
    uint16_t LHCMaskLowTop;

    uint16_t LHCMaskHighBottom;
    uint16_t LHCMaskHighTop;

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


  struct LUMI_SUMMARY {

    float DeadTimeNormalization; 
    float LHCNormalization; // recieved from LHC 

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

    float LHCLumi[HCAL_HLX_MAX_BUNCHES]; // Sum of LHC.data over all HLX's

    float ETLumi[HCAL_HLX_MAX_BUNCHES];
    float ETLumiErr[HCAL_HLX_MAX_BUNCHES];
    int16_t ETLumiQlty[HCAL_HLX_MAX_BUNCHES];
    float ETBXNormalization[HCAL_HLX_MAX_BUNCHES];

    float OccLumi[2][HCAL_HLX_MAX_BUNCHES];
    float OccLumiErr[2][HCAL_HLX_MAX_BUNCHES];
    int16_t OccLumiQlty[2][HCAL_HLX_MAX_BUNCHES];
    float OccBXNormalization[2][HCAL_HLX_MAX_BUNCHES];
  };

  /**************** Trigger *********************/


  struct LEVEL1_PATH {

    char pathName[128];
    uint64_t counts;
    uint64_t prescale;
    uint64_t deadtimecount;
  };

  struct LEVEL1_TRIGGER {
    uint32_t runNumber;
    uint32_t sectionNumber; // Lumi section number recorded by the daq.

    char GTLumiInfoFormat[32];
   
    LEVEL1_PATH GTAlgo[128];
    LEVEL1_PATH GTTech[64];

    uint32_t GTPartition0TriggerCounts[10];
    uint32_t GTPartition0DeadTime[10];
  };

  struct HLT_PATH {  // only object that uses STL and is variable size.
    char PathName[128]; //This is the name of trigger path
    uint32_t L1Pass;      //Number of times the path was entered
    uint32_t PSPass;      //Number after prescaling
    uint32_t PAccept;     //Number of accepts by the trigger path
    uint32_t PExcept;     //Number of exceptional event encountered
    uint32_t PReject;     
    char PrescalerModule[64]; //Name of the prescale module in the path
    uint32_t PSIndex;     //Index into the set of pre defined prescales
    uint32_t Prescale;
  };

  struct HLTRIGGER {
    uint32_t runNumber;
    uint32_t sectionNumber;

    HLT_PATH HLTPaths[256];
  };

  /***************** Internal use ****************/
  struct LUMI_RAW_HEADER { // Used in NibbleCollector 
    uint16_t marker;
    uint8_t  hlxID;
    uint8_t  packetID;
    uint32_t startOrbit;
    uint16_t numOrbits;
    uint16_t startBunch; // Starting bunch in this packet
    uint16_t numBunches; // Total number of bunches in histogram
    uint8_t  histogramSet;
    uint8_t  histogramSequence;
    uint16_t allA;
    uint16_t allF;
  };

  struct LUMI_NIBBLE_HEADER {
    uint32_t  startOrbit;
    uint16_t  numOrbits;
    uint16_t  numBunches; // Number of bunches histogrammed
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


  struct LUMI_SECTION_HEADER {
    uint32_t timestamp;
    uint32_t timestamp_micros;

    uint32_t  runNumber;   // Run number
    uint32_t  sectionNumber; // Section number

    uint32_t  startOrbit;  // Start orbit of lumi section
    uint32_t  numOrbits;   // Total number of orbits recorded in lumi section
    uint16_t  numBunches;  // Total number of bunches (from start of orbit)
    uint16_t  numHLXs;     // Number of HLXs in lumi section

    bool bCMSLive;    // Is CMS taking data?
    bool bOC0;        // Was section initialised by an OC0?
  };

  struct LUMI_SECTION_SUB_HEADER {
    uint32_t  numNibbles;  // Number of nibbles in this histogram
    bool bIsComplete; // Is this histogram complete (i.e. no missing nibbles)
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
    LUMI_DETAIL  lumiDetail;

    ET_SUM_SECTION etSum[HCAL_HLX_MAX_HLXS];
    OCCUPANCY_SECTION occupancy[HCAL_HLX_MAX_HLXS];
    LHC_SECTION lhc[HCAL_HLX_MAX_HLXS];
  };
}//~namespace lumi

#endif //~LUMIDATASTRUCTURES_H

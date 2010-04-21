#ifndef RecoLuminosity_LumiProducer_LUMIRAWDATASTRUCTURES_H
#define RecoLuminosity_LumiProducer_LUMIRAWDATASTRUCTURES_H
//Note: this header file corresponds to svn.cern.ch/reps/Luminosity/HLXReadOut/HLXCoreLibs/include/LumiStructures.hh - Revision 124 LumiDAQ release-2.2
/*
  LumiNibble structure definitions
*/

// The string and stream definitions
// Type definitions used by the HAL, etc...

#define HCAL_HLX_MAX_BUNCHES 4096
#define HCAL_HLX_MAX_HLXS 36

// Changes
// Namespace for the HCAL HLX
#include <stdint.h>

namespace HCAL_HLX{

  struct DAQ_HEART_BEAT {

    uint32_t runNumber;
    uint32_t sectionNumber;
    uint32_t bCMSLive;

    uint32_t timestamp;
    uint32_t timestamp_micros;
  };

  struct RUN_SUMMARY {

    char runSequenceName[128];
    uint32_t HLTConfigId;
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
    uint32_t runNumber;
    uint32_t sectionNumber;

    int HLX;
    int HFLumi;
    int ECAL;
    int HCAL;
    int Tracker;
    int RPC;
    int DT;
    int CSC;
  };

  struct RCMS_CONFIG {

    uint32_t runNumber;
    
    char CMSSW_Tag[32];
    char TriDAS_Tag[32];
    
    uint32_t FirmwareVersion;
    uint32_t ExpectedFirmwareVersion;
    char AddressTablePath[256];
    
    char CfgDBTag[64];
    char CfgDBAccessor[64];
    bool UseConfigDB;
    
    uint32_t DestIPHigh;
    uint32_t DestIPLow;
    
    uint32_t SrcIPBaseHigh;
    uint32_t SrcIPBaseLow;
    
    uint32_t DestMacAddrHigh;
    uint32_t DestMacAddrMed;
    uint32_t DestMacAddrLow;
    
    uint32_t SrcMacAddrHigh;
    uint32_t SrcMacAddrMed;
    uint32_t SrcMacAddrLow;
    
    uint32_t SrcPort;
    uint32_t DestPort;
    
    uint32_t DebugData;
    uint32_t DebugReadout;
    uint32_t DebugSingleCycle;
    
    uint32_t NumOrbits;
    uint32_t OrbitPeriod;
    
    uint32_t Id;
    
    uint32_t TTCBC0Pattern;
    uint32_t TTCSTARTPattern;
    uint32_t TTCSTOPPattern;
    
    uint32_t BC0Delay;
    
    uint32_t OccThresholdLowBottom;
    uint32_t OccThresholdLowTop;
    uint32_t OccThresholdHighBottom;
    uint32_t OccThresholdHighTop;
    
    uint32_t LHCThresholdBottom;
    uint32_t LHCThresholdTop;
    
    uint32_t ETSumCutoffBottom;
    uint32_t ETSumCutoffTop;
    
    uint32_t OCCMaskBottom;
    uint32_t OCCMaskTop;
    
    uint32_t LHCMaskLowBottom;
    uint32_t LHCMaskLowTop;
    uint32_t LHCMaskHighBottom;
    uint32_t LHCMaskHighTop;
    
    uint32_t SumETMaskLowBottom;
    uint32_t SumETMaskLowTop;
    uint32_t SumETMaskHighBottom;
    uint32_t SumETMaskHighTop;
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
  };

  struct LEVEL1_TRIGGER {
    uint32_t runNumber;
    uint32_t sectionNumber; // Lumi section number recorded by the daq.

    uint32_t timestamp;
    uint32_t timestamp_micros;

    uint64_t deadtimecount;

    char GTLumiInfoFormat[32];
   
    LEVEL1_PATH GTAlgo[128];
    LEVEL1_PATH GTTech[64];
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

    uint32_t HLTConfigId;
  };

  struct HLTRIGGER {
    uint32_t runNumber;
    uint32_t sectionNumber;
    uint32_t numPaths;

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

  //***********************************************************

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

  struct DIP_STRUCT_BASE { 
    int MessageQuality;     
    uint32_t timestamp;
    uint32_t timestamp_micros;
    uint32_t runNumber;
    uint32_t sectionNumber;
  };
  
    struct VDM_SCAN_DATA: public DIP_STRUCT_BASE {
    bool VdmMode; //True when a scan at one of the IPs is imminent, false otherwise
    int IP;
    bool RecordDataFlag; // True while data for one of the scan points at one of the IPs is being taken, false otherwise   
    double BeamSeparation; //separation in sigma for the scan point  
    bool isXaxis; //true if scanning xaxis, otherwise yaxis is being scanned 
    int Beam; 
    double StepProgress;
  };

  struct BRAN_DATA: public DIP_STRUCT_BASE {
    double MeanCrossingAngle;
    int AcqMode; 
    double MeanLuminosity;
 };

  struct BRAN_BX_DATA: public DIP_STRUCT_BASE {
    double bunchByBunchLuminosity[3564];
    int AcqMode;
  };

  struct LHC_FILL_DATA: public DIP_STRUCT_BASE {
    uint32_t FillNumber;
  };


  struct CMS_LUMI_DIP_DATA: public DIP_STRUCT_BASE {
    uint32_t numHLXs;
    uint32_t startOrbit;
    uint32_t numOrbits;
    uint32_t numBunches;
    float instantLumi;
    float instantLumiErr;
  };

  struct CMS_LUMI_LH_DIP_DATA: public DIP_STRUCT_BASE {
    float lumiHisto[3564];
    uint32_t numBunches;
  };

  struct CMS_STATUS_DATA: public DIP_STRUCT_BASE {
    char status[64];
  };

  struct TRIGGER_LUMI_SEGMENT: public DIP_STRUCT_BASE {
    char state[64];
    uint32_t deadtime;
    uint32_t deadtimeBeamActive;
    uint32_t lumiSegmentNumber;
    
  };

  struct LHC_BEAM_MODE_DATA: public DIP_STRUCT_BASE {
    char beamMode[64];
  };

  struct LHC_BEAM_ENERGY_DATA: public DIP_STRUCT_BASE {
    float singleBeamEnergy; //GeV
  };

  struct LHC_BEAM_INTENSITY_DATA: public DIP_STRUCT_BASE {
    double beamIntensity; //total num protons in beam summed over all bxs
    double primitiveLifetime;
    uint64_t acqTimeStamp;
  };

  struct LHC_BEAM_FBCT_INTENSITY_DATA: public DIP_STRUCT_BASE {
    double bestLifetime; 
    double averageBeamIntensity;
    float averageBunchIntensities[3564];
  };

  struct CMS_SCAN_TUNE_DATA: public DIP_STRUCT_BASE {
    double IntTime; 
    double CollRate;
    double CollRateErr;
    bool Preferred;
    char Source[64];
  };

  struct DIP_ACQUISITION_MODE: public DIP_STRUCT_BASE {
    char mode[64];
  };

  struct BEAM_INFO {
    double totalIntensity;
    double primitiveLifetime;
    double bestLifeTime;
    double averageBeamIntensity;
    float orbitFrequency;
    float averageBunchIntensities[3564];
  };
  
  
  struct BRANA_INFO {
    double meanCrossingAngle;
    int acqMode;
    double meanLuminosity;
    double bunchByBunchLuminosity[3564];
  };

  struct BRANP_INFO {
    double meanCrossingAngle;
    int acqMode;
    int counterAcquisition;
    double meanLuminosity;
    double meanCrossingAngleError;
    double meanLuminosityError;
    double bunchByBunchLuminosity[3564];
  };

  struct BRAN_INFO {

    BRANA_INFO branA;
    BRANP_INFO branP;
  };

  struct DIP_COMBINED_DATA: public DIP_STRUCT_BASE {

     char beamMode[128];

     float Energy;

    uint32_t FillNumber;

     BEAM_INFO Beam[2];

     BRAN_INFO BRAN4L1;
     BRAN_INFO BRAN4R1;
     BRAN_INFO BRAN4L5;
     BRAN_INFO BRAN4R5;

     VDM_SCAN_DATA VdMScan;

  };

}//~namespace HCAL_HLX

#endif //RecoLuminosity_LumiProducer_LUMIRAWDATASTRUCTURES_H

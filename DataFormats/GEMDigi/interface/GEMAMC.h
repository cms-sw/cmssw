#ifndef DataFormats_GEMDigi_GEMAMC_h
#define DataFormats_GEMDigi_GEMAMC_h
#include "GEMOptoHybrid.h"
#include <vector>

class GEMAMC {
public:
  union AMCheader1 {
    uint64_t word;
    struct {
      uint64_t dataLength : 20;  // Always 0xfffff, use trailer dataLengthT
      uint64_t bxID : 12;        // Bunch crossing ID
      uint64_t l1AID : 24;       // L1A number – basically this is like event number, but reset by resync
      uint64_t AMCnum : 4;       // Slot number of the AMC
      uint64_t reserved : 4;     // not used
    };
  };
  union AMCheader2 {
    uint64_t word;
    // v301 dataformat
    struct {
      uint64_t boardID : 16;   // 8bit long GLIB serial number
      uint64_t orbitNum : 16;  // Orbit number, Reset by EC0
      uint64_t param3 : 8;     // RunPar - Controlled by software, normally used only for calibrations
      uint64_t param2 : 8;     // RunPar - Controlled by software, normally used only for calibrations
      uint64_t param1 : 8;     // RunPar - Controlled by software, normally used only for calibrations
      uint64_t runType : 4;    // run types like physics, cosmics, threshold scan, latency scan, etc..
      uint64_t formatVer : 4;  // Current format version = 0x0
    };
    // v302 dataformat
    struct {
      uint64_t softSrcId : 12;     // FED ID - Configruation Error if does not match with CDF header
      uint64_t softSlot : 4;       // AMC slot number - Configuation Error if does not match with AMC13 AHn header
      uint64_t orbitNumV302 : 32;  // Orbit counter, Reset by EC0 - Error if does not match AMC BH header
      uint64_t : 12;               // unused
      uint64_t FVv302 : 4;         // Current version = 0x1
    };
  };
  union AMCTrailer {
    uint64_t word;
    struct {
      uint64_t dataLength : 20;  // Number of 64bit words in this event
      uint64_t : 4;              // unused
      uint64_t l1AID : 8;        // L1A number (first 8 bits)
      uint64_t crc : 32;         // CRC added by the AMC13
    };
  };
  union EventHeader {
    uint64_t word;
    struct {
      uint64_t ttsState : 4;  // GLIB TTS state at the moment when this event was built.
      uint64_t pType : 4;     // Payload type: can be one refering to different zero suppression schemes
      // in normal data taking or calibration type.
      // Note in calibration type the entire GCT (GEM Chamber Trailer) is skipped
      uint64_t pVer : 3;        // Version of the payload type
      uint64_t davCnt : 5;      // Number of chamber blocks in this event
      uint64_t buffState : 24;  // Buffer status, Always 0 in current fw
      uint64_t davList : 24;    // Data Available list: a bitmask indicating which chambers have data in this event
    };
  };
  union EventTrailer {
    uint64_t word;
    // v301 dataformat
    struct {
      uint64_t BCL : 4;  // 1st bit, BC0 locked - If 0, this is a bad condition indicating a
      // problem in the clock or TTC command stream (critical condition)
      uint64_t DR : 1;        // DAQ Ready - If 0, this means that AMC13 is not ready to take data (critical condition)
      uint64_t CL : 1;        // DAQ clock locked- If 0, this indicates a problem in the DAQ clock (critical condition)
      uint64_t ML : 1;        // MMCM locked - Should always be 1
      uint64_t BP : 1;        // Backpressure - If this is 1, it means that we are receiving backpressure from AMC13
      uint64_t oosGlib : 32;  // GLIB is out‐of‐sync (critical): L1A ID is different for
      // different chambers in this event (1 bit)
      uint64_t linkTo : 24;  // Link timeout flags (one bit for each link indicating timeout condition)
    };
    // v302 dataformat
    struct {
      uint64_t L1aNF : 1;    // L1A FIFO near full - Warning
      uint64_t L1aF : 1;     // L1A FIFO full - Error
      uint64_t : 1;          // unused
      uint64_t BCLv302 : 1;  // BC0 locked - If 0, this is a bad condition indicating a
      // problem in the clock or TTC command stream (critical condition)
      uint64_t DRv302 : 1;   // DAQ Ready - If 0, this means that AMC13 is not ready to take data (critical condition)
      uint64_t CLv302 : 1;   // DAQ clock locked- If 0, this indicates a problem in the DAQ clock (critical condition)
      uint64_t MLv302 : 1;   // MMCM locked - Should always be 1
      uint64_t BPv302 : 1;   // Backpressure - If this is 1, it means that we are receiving backpressure from AMC13
      uint64_t param3 : 8;   // RunPar - Controlled by software, normally used only for calibrations
      uint64_t param2 : 8;   // RunPar - Controlled by software, normally used only for calibrations
      uint64_t param1 : 8;   // RunPar - Controlled by software, normally used only for calibrations
      uint64_t runType : 4;  // Type of Run - Controlled by software, “physics” is assigned 0x1,
      // hits from events with other run types should be discarded
      uint64_t : 4;              // unused
      uint64_t linkToV302 : 24;  // Link timeout flags (one bit for each link indicating timeout condition)
    };
  };

  GEMAMC() : amch1_(0), amch2_(0), amct_(0), eh_(0), et_(0){};
  ~GEMAMC() { gebd_.clear(); }

  int status();

  void setAMCheader1(uint64_t word) { amch1_ = word; }
  void setAMCheader1(uint32_t dataLength, uint16_t bxID, uint32_t l1AID, uint8_t AMCnum);
  uint64_t getAMCheader1() const { return amch1_; }

  void setAMCheader2(uint64_t word) { amch2_ = word; }
  void setAMCheader2(uint16_t boardID, uint16_t orbitNum, uint8_t runType);
  uint64_t getAMCheader2() const { return amch2_; }

  void setAMCTrailer(uint64_t word) { amct_ = word; }
  uint64_t getAMCTrailer() const { return amct_; }

  void setGEMeventHeader(uint64_t word) { eh_ = word; }
  void setGEMeventHeader(uint8_t davCnt, uint32_t davList);
  uint64_t getGEMeventHeader() const { return eh_; }

  void setGEMeventTrailer(uint64_t word) { et_ = word; }
  uint64_t getGEMeventTrailer() const { return et_; }

  // v301
  uint32_t dataLength() const { return AMCTrailer{amct_}.dataLength; }
  uint16_t bunchCrossing() const { return AMCheader1{amch1_}.bxID; }
  uint32_t lv1Id() const { return AMCheader1{amch1_}.l1AID; }
  uint8_t amcNum() const { return AMCheader1{amch1_}.AMCnum; }

  uint16_t boardId() const { return AMCheader2{amch2_}.boardID; }
  uint32_t orbitNumber() const {
    if (formatVer() == 0)
      return AMCheader2{amch2_}.orbitNum;
    return AMCheader2{amch2_}.orbitNumV302;
  }
  uint8_t param3() const {
    if (formatVer() == 0)
      return AMCheader2{amch2_}.param3;
    return EventTrailer{et_}.param3;
  }
  uint8_t param2() const {
    if (formatVer() == 0)
      return AMCheader2{amch2_}.param2;
    return EventTrailer{et_}.param2;
  }
  uint8_t param1() const {
    if (formatVer() == 0)
      return AMCheader2{amch2_}.param1;
    return EventTrailer{et_}.param1;
  }
  uint8_t runType() const {
    if (formatVer() == 0)
      return AMCheader2{amch2_}.runType;
    return EventTrailer{et_}.runType;
  }
  // SAME in V301 and V302
  uint8_t formatVer() const { return AMCheader2{amch2_}.formatVer; }

  uint8_t lv1Idt() const { return AMCTrailer{amct_}.l1AID; }
  uint32_t crc() const { return AMCTrailer{amct_}.crc; }

  uint16_t ttsState() const { return EventHeader{eh_}.ttsState; }
  uint8_t davCnt() const { return EventHeader{eh_}.davCnt; }
  uint32_t buffState() const { return EventHeader{eh_}.buffState; }
  uint32_t davList() const { return EventHeader{eh_}.davList; }

  uint8_t bc0locked() const { return EventTrailer{et_}.BCL; }
  uint8_t daqReady() const { return EventTrailer{et_}.DR; }
  uint8_t daqClockLocked() const { return EventTrailer{et_}.CL; }
  uint8_t mmcmLocked() const { return EventTrailer{et_}.ML; }
  uint8_t backPressure() const { return EventTrailer{et_}.BP; }
  uint8_t oosGlib() const { return EventTrailer{et_}.oosGlib; }
  uint32_t linkTo() const { return EventTrailer{et_}.linkTo; }

  // v302
  uint16_t softSrcId() const { return AMCheader2{amch2_}.softSrcId; }
  uint8_t softSlot() const { return AMCheader2{amch2_}.softSlot; }

  uint8_t l1aNF() const { return EventTrailer{et_}.L1aNF; }
  uint8_t l1aF() const { return EventTrailer{et_}.L1aF; }

  //!Adds GEB data to vector
  void addGEB(GEMOptoHybrid g) { gebd_.push_back(g); }
  //!Returns a vector of GEB data
  const std::vector<GEMOptoHybrid>* gebs() const { return &gebd_; }
  //!Clear a vector of GEB data
  void clearGEBs() { gebd_.clear(); }

private:
  uint64_t amch1_;
  uint64_t amch2_;
  uint64_t amct_;
  uint64_t eh_;
  uint64_t et_;

  std::vector<GEMOptoHybrid> gebd_;  ///<Vector of GEB data
};
#endif

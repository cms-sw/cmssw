#ifndef DATAFORMATS_TCDS_TCDSRECORD_H
#define DATAFORMATS_TCDS_TCDSRECORD_H

//---------------------------------------------------------------------------
//!  \class TCDSRecord
//!  \brief Class to contain information from TCDS FED
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------

#include <bitset>
#include <ostream>
#include <cstdint>
#include <vector>

#include "DataFormats/TCDS/interface/BSTRecord.h"
#include "DataFormats/TCDS/interface/L1aInfo.h"

class TCDSRecord {
public:
  enum BGo {
    LumiNibble = 0,
    BC0 = 1,
    TestEnable = 2,
    PrivateGap = 3,
    PrivateOrbit = 4,
    Resync = 5,
    HardReset = 6,
    EC0 = 7,
    OC0 = 8,
    Start = 9,
    Stop = 10,
    StartOfGap = 11,
    WarningTestEnable = 13
  };

  enum BSTstatus {
    Unknown = 0x00000000,
    Reset = 0x0000dead,
    Unlocked = 0xfa11010c,
    NoData = 0xfa110acc,
    Okay = 0x0000bea0
  };

  TCDSRecord();
  TCDSRecord(const unsigned char* rawData);
  virtual ~TCDSRecord();

  // MAC address of the CPM
  uint64_t getMacAddress() const { return macAddress_; }

  // Software version of TCDS s/w
  uint32_t getSwVersion() const { return swVersion_; }

  // Firmware version of the CPM
  uint32_t getFwVersion() const { return fwVersion_; }

  // Version of the TCDS record
  uint32_t getRecordVersion() const { return recordVersion_; }

  // Run number
  uint32_t getRunNumber() const { return runNumber_; }

  // Lumi section number
  uint32_t getLumiSection() const { return lumiSection_; }

  // Lumi nibble number
  uint32_t getNibble() const { return nibble_; }

  // Number of nibbles per lumi section
  uint16_t getNibblesPerLumiSection() const { return nibblesPerLumiSection_; }

  // The event type corresponding to edm::EventAuxiliary::ExperimentType
  uint16_t getEventType() const { return eventType_; }

  // The trigger word contains sixteen boolean flags corresponding to the sixteen trigger types
  // (see https://twiki.cern.ch/twiki/bin/view/CMS/TcdsEventRecord#TCDS_Event_Trigger_Type_Definiti)
  // If a given trigger type fired for this event, the corresponding flag will be true.
  uint16_t getTriggerTypeFlags() const { return triggerTypeFlags_; }

  // Input state at Triggered BX +/- 3, currently zeros
  uint16_t getInputs() const { return inputs_; }

  // Bunch-crossing identified
  uint16_t getBXID() const { return bxid_; }

  // Orbit number
  uint64_t getOrbitNr() const { return orbitNr_; }

  // Number of triggers since last EC0
  uint64_t getTriggerCount() const { return triggerCount_; }

  // Number of events since start of the run (last OC0)
  uint64_t getEventNumber() const { return eventNumber_; }

  // BST reception status corresponding to TCDSRecord::BSTstatus
  uint32_t getBstReceptionStatus() const { return bstReceptionStatus_; }

  // The BST message as received from the LHC
  const BSTRecord& getBST() const { return bst_; }

  // List of active paritions, currently not implemented
  typedef std::bitset<96> ActivePartitions;
  ActivePartitions getActivePartitions() const { return activePartitions_; }

  // History of recent L1 accepts
  typedef std::vector<L1aInfo> L1aHistory;
  const L1aHistory& getFullL1aHistory() const { return l1aHistory_; }
  const L1aInfo& getL1aHistoryEntry(const uint8_t entry) const { return l1aHistory_.at(entry); }

  // Orbit number when the given Bgo was sent last
  uint32_t getOrbitOfLastBgo(const uint16_t bgo) const { return lastBgos_.at(bgo); }

  // Orbit number of last OC0
  uint32_t getLastOrbitCounter0() const { return lastBgos_.at(BGo::OC0); }

  // Orbit number of last Test Enable
  uint32_t getLastTestEnable() const { return lastBgos_.at(BGo::TestEnable); }

  // Orbit number of last Resync
  uint32_t getLastResync() const { return lastBgos_.at(BGo::Resync); }

  // Orbit number of last Start
  uint32_t getLastStart() const { return lastBgos_.at(BGo::Start); }

  // Orbit number of last EC0
  uint32_t getLastEventCounter0() const { return lastBgos_.at(BGo::EC0); }

  // Orbit number of last Hard Reset
  uint32_t getLastHardReset() const { return lastBgos_.at(BGo::HardReset); }

private:
  uint64_t orbitNr_;
  uint64_t triggerCount_;
  uint64_t eventNumber_;
  uint64_t macAddress_;
  uint32_t swVersion_;
  uint32_t fwVersion_;
  uint32_t recordVersion_;
  uint32_t runNumber_;
  uint32_t bstReceptionStatus_;
  uint32_t nibble_;
  uint32_t lumiSection_;
  uint16_t nibblesPerLumiSection_;
  uint16_t eventType_;
  uint16_t triggerTypeFlags_;
  uint16_t inputs_;
  uint16_t bxid_;

  ActivePartitions activePartitions_;
  L1aHistory l1aHistory_;

  BSTRecord bst_;

  std::vector<uint32_t> lastBgos_;
};

/// Pretty-print operator for TCDSRecord
std::ostream& operator<<(std::ostream&, const TCDSRecord&);

#endif  // DATAFORMATS_TCDS_TCDSRECORD_H

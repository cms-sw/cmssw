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


class TCDSRecord
{
public:

  enum BGo {
    LumiNibble        = 0,
    BC0               = 1,
    TestEnable        = 2,
    PrivateGap        = 3,
    PrivateOrbit      = 4,
    Resync            = 5,
    HardReset         = 6,
    EC0               = 7,
    OC0               = 8,
    Start             = 9,
    Stop              = 10,
    StartOfGap        = 11,
    WarningTestEnable = 13
  };

  TCDSRecord();
  TCDSRecord(const unsigned char* rawData);
  virtual ~TCDSRecord();

  uint16_t getEventType() const { return eventType_; }
  uint64_t getMacAddress() const { return macAddress_; }
  uint32_t getSwVersion() const { return swVersion_; }
  uint32_t getFwVersion() const { return fwVersion_; }
  uint32_t getRecordVersion() const { return recordVersion_; }
  uint32_t getRunNumber() const { return runNumber_; }
  uint32_t getBstReceptionStatus() const { return bstReceptionStatus_; }
  uint32_t getNibble() const { return nibble_; }
  uint32_t getLumiSection() const { return lumiSection_; }
  uint16_t getNibblesPerLumiSection() const { return nibblesPerLumiSection_; }
  uint16_t getTriggerTypeFlags() const { return triggerTypeFlags_; }
  uint16_t getInputs() const { return inputs_; }
  uint16_t getBXID() const { return bxid_; }
  uint64_t getOrbitNr() const { return orbitNr_; }
  uint64_t getTriggerCount() const { return triggerCount_; }
  uint64_t getEventNumber() const { return eventNumber_; }

  typedef std::bitset<96> ActivePartitions;
  ActivePartitions getActivePartitions() const { return activePartitions_; }

  typedef std::vector<L1aInfo> L1aHistory;
  const L1aHistory& getFullL1aHistory() const { return l1aHistory_; }

  const BSTRecord& getBST() const { return bst_; }

  const L1aInfo& getL1aHistoryEntry(const uint8_t entry) const { return l1aHistory_.at(entry); }

  uint32_t getOrbitOfLastBgo(const uint16_t bgo) const { return lastBgos_.at(bgo); }
  uint32_t getLastOrbitCounter0() const { return lastBgos_.at(BGo::OC0); }
  uint32_t getLastTestEnable() const { return lastBgos_.at(BGo::TestEnable); }
  uint32_t getLastResync() const { return lastBgos_.at(BGo::Resync); }
  uint32_t getLastStart() const { return lastBgos_.at(BGo::Start); }
  uint32_t getLastEventCounter0() const { return lastBgos_.at(BGo::EC0); }
  uint32_t getLastHardReset() const { return lastBgos_.at(BGo::HardReset); }


private:

  uint16_t eventType_;
  uint64_t macAddress_;
  uint32_t swVersion_;
  uint32_t fwVersion_;
  uint32_t recordVersion_;
  uint32_t runNumber_;
  uint32_t bstReceptionStatus_;
  uint32_t nibble_;
  uint32_t lumiSection_;
  uint16_t nibblesPerLumiSection_;
  uint16_t triggerTypeFlags_;
  uint16_t inputs_;
  uint16_t bxid_;
  uint64_t orbitNr_;
  uint64_t triggerCount_;
  uint64_t eventNumber_;

  ActivePartitions activePartitions_;
  L1aHistory l1aHistory_;

  BSTRecord bst_;

  std::vector<uint32_t> lastBgos_;

};

/// Pretty-print operator for TCDSRecord
std::ostream& operator<<(std::ostream&, const TCDSRecord&);

#endif // DATAFORMATS_TCDS_TCDSRECORD_H

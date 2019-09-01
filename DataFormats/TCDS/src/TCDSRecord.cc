#include <iomanip>

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"

TCDSRecord::TCDSRecord()
    : orbitNr_(0),
      triggerCount_(0),
      eventNumber_(0),
      macAddress_(0),
      swVersion_(0),
      fwVersion_(0),
      recordVersion_(0),
      runNumber_(0),
      bstReceptionStatus_(0),
      nibble_(0),
      lumiSection_(0),
      nibblesPerLumiSection_(0),
      eventType_(0),
      triggerTypeFlags_(0),
      inputs_(0),
      bxid_(0) {}

TCDSRecord::TCDSRecord(const unsigned char* rawData) {
  tcds::Raw_v1 const* tcdsRaw = reinterpret_cast<tcds::Raw_v1 const*>(rawData + FEDHeader::length);
  const FEDHeader fedHeader(rawData);

  orbitNr_ = (tcdsRaw->header.orbitHigh << 16) | tcdsRaw->header.orbitLow;
  triggerCount_ = tcdsRaw->header.triggerCount;
  eventNumber_ = tcdsRaw->header.eventNumber;
  macAddress_ = tcdsRaw->header.macAddress;
  swVersion_ = tcdsRaw->header.swVersion;
  fwVersion_ = tcdsRaw->header.fwVersion;
  recordVersion_ = tcdsRaw->header.recordVersion;
  runNumber_ = tcdsRaw->header.runNumber;
  bstReceptionStatus_ = tcdsRaw->header.bstReceptionStatus;
  nibble_ = tcdsRaw->header.nibble;
  lumiSection_ = tcdsRaw->header.lumiSection;
  nibblesPerLumiSection_ = tcdsRaw->header.nibblesPerLumiSection;
  eventType_ = fedHeader.triggerType();
  triggerTypeFlags_ = tcdsRaw->header.triggerTypeFlags;
  inputs_ = tcdsRaw->header.inputs;
  bxid_ = tcdsRaw->header.bxid;

  activePartitions_ = ActivePartitions(tcdsRaw->header.activePartitions0);
  activePartitions_ |= ActivePartitions(tcdsRaw->header.activePartitions1) << 32;
  activePartitions_ |= ActivePartitions(tcdsRaw->header.activePartitions2) << 64;

  bst_ = BSTRecord(tcdsRaw->bst);

  for (auto i = 0; i < tcds::l1aHistoryDepth_v1; ++i) {
    l1aHistory_.emplace_back(L1aInfo(tcdsRaw->l1aHistory.l1aInfo[i]));
  }

  for (auto i = 0; i < tcds::bgoCount_v1; ++i) {
    lastBgos_.emplace_back(((uint64_t)(tcdsRaw->bgoHistory.lastBGo[i].orbithigh) << 32) |
                           tcdsRaw->bgoHistory.lastBGo[i].orbitlow);
  }
}

TCDSRecord::~TCDSRecord() {}

std::ostream& operator<<(std::ostream& s, const TCDSRecord& record) {
  s << "MacAddress:            0x" << std::hex << record.getMacAddress() << std::dec << std::endl;
  s << "SwVersion:             0x" << std::hex << record.getSwVersion() << std::dec << std::endl;
  s << "FwVersion:             0x" << std::hex << record.getFwVersion() << std::dec << std::endl;
  s << "RecordVersion:         " << record.getRecordVersion() << std::endl;
  s << "RunNumber:             " << record.getRunNumber() << std::endl;
  s << "BstReceptionStatus:    0x" << std::hex << record.getBstReceptionStatus() << std::dec << std::endl;
  s << "Nibble:                " << record.getNibble() << std::endl;
  s << "LumiSection:           " << record.getLumiSection() << std::endl;
  s << "NibblesPerLumiSection: " << record.getNibblesPerLumiSection() << std::endl;
  s << "EventType:             " << record.getEventType() << std::endl;
  s << "TriggerTypeFlags:      0x" << std::hex << record.getTriggerTypeFlags() << std::dec << std::endl;
  s << "Inputs:                " << record.getInputs() << std::endl;
  s << "OrbitNr:               " << record.getOrbitNr() << std::endl;
  s << "BXID:                  " << record.getBXID() << std::endl;
  s << "TriggerCount:          " << record.getTriggerCount() << std::endl;
  s << "EventNumber:           " << record.getEventNumber() << std::endl;
  s << "ActivePartitions:      " << record.getActivePartitions() << std::endl;
  s << std::endl;

  s << "L1aHistory:" << std::endl;
  for (auto l1Info : record.getFullL1aHistory())
    s << l1Info;
  s << std::endl;

  s << record.getBST() << std::endl;
  s << "LastOrbitCounter0:     " << record.getLastOrbitCounter0() << std::endl;
  s << "LastTestEnable:        " << record.getLastTestEnable() << std::endl;
  s << "LastResync:            " << record.getLastResync() << std::endl;
  s << "LastStart:             " << record.getLastStart() << std::endl;
  s << "LastEventCounter0:     " << record.getLastEventCounter0() << std::endl;
  s << "LastHardReset:         " << record.getLastHardReset() << std::endl;

  for (auto i = 0; i < tcds::bgoCount_v1; ++i)
    s << "Last BGo " << std::setw(2) << i << ": " << record.getOrbitOfLastBgo(i) << std::endl;

  return s;
}

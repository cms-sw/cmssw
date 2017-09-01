#include "DataFormats/TCDS/interface/TCDSRecord.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"


TCDSRecord::TCDSRecord() :
  macAddress_(0),
  swVersion_(0),
  fwVersion_(0),
  recordVersion_(0),
  runNumber_(0),
  nibble_(0),
  lumiSection_(0),
  nibblesPerLumiSection_(0),
  triggerTypeFlags_(0),
  inputs_(0),
  bcid_(0),
  orbit_(0),
  triggerCount_(0),
  eventNumber_(0)
{}


TCDSRecord::TCDSRecord(const unsigned char* rawData)
{
  tcds::Raw_v1 const* tcdsRaw =
    reinterpret_cast<tcds::Raw_v1 const*>(rawData);

  macAddress_ = tcdsRaw->header.macAddress;
  swVersion_ = tcdsRaw->header.swVersion;
  fwVersion_ = tcdsRaw->header.fwVersion;
  recordVersion_ = tcdsRaw->header.recordVersion;
  runNumber_ = tcdsRaw->header.runNumber;
  bstReceptionStatus_ = tcdsRaw->header.bstReceptionStatus;
  nibble_ = tcdsRaw->header.nibble;
  lumiSection_ = tcdsRaw->header.lumiSection;
  nibblesPerLumiSection_ = tcdsRaw->header.nibblesPerLumiSection;
  triggerTypeFlags_ = tcdsRaw->header.triggerTypeFlags;
  inputs_ = tcdsRaw->header.inputs;
  bcid_ = tcdsRaw->header.bcid;
  orbit_ = (tcdsRaw->header.orbitHigh << 16) | tcdsRaw->header.orbitLow;
  triggerCount_ = tcdsRaw->header.triggerCount;
  eventNumber_ = tcdsRaw->header.eventNumber;

  activePartitions_  = ActivePartitions(tcdsRaw->header.activePartitions0);
  activePartitions_ |= ActivePartitions(tcdsRaw->header.activePartitions1) << 32;
  activePartitions_ |= ActivePartitions(tcdsRaw->header.activePartitions2) << 64;

  bst_ = BSTRecord(tcdsRaw->bst);

  for (auto i = 0; i < tcds::l1aHistoryDepth_v1; ++i)
  {
    l1aHistory_.emplace_back(L1aInfo(tcdsRaw->l1aHistory.l1aInfo[i]));
  }

  for (auto i = 0; i < tcds::bgoCount_v1; ++i)
  {
    lastBgos_.emplace_back(tcdsRaw->bgoHistory.lastBGo[i].lastOrbit);
  }
}


TCDSRecord::~TCDSRecord() {}

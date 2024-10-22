#include <iomanip>
#include <ctime>

#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"

const DCSRecord::ParitionNames DCSRecord::partitionNames_ = {
    {"EBp",  "EBm",    "EEp", "EEm",    "HBHEa", "HBHEb", "HBHEc", "HF",   "HO",   "RPC", "DT0", "DTp",  "DTm", "CSCp",
     "CSCm", "CASTOR", "ZDC", "TIBTID", "TOB",   "TECp",  "TECm",  "BPIX", "FPIX", "ESp", "ESm", "GEMp", "GEMm"}};

DCSRecord::DCSRecord() : timestamp_(edm::Timestamp::invalidTimestamp()), magnetCurrent_(-1) {}

DCSRecord::DCSRecord(const online::DCS_v1& dcs) {
  // DIP timestamp is in milliseconds
  const uint64_t seconds = dcs.timestamp / 1000;
  const uint32_t microseconds = (dcs.timestamp % 1000) * 1000;
  timestamp_ = edm::Timestamp((seconds << 32) | microseconds);
  highVoltageReady_ = dcs.highVoltageReady;
  //bit always valid for V1
  highVoltageValid_ = 0xffffffff;
  magnetCurrent_ = dcs.magnetCurrent;
}

DCSRecord::DCSRecord(const online::DCS_v2& dcs) {
  // DIP timestamp is in milliseconds
  const uint64_t seconds = dcs.timestamp / 1000;
  const uint32_t microseconds = (dcs.timestamp % 1000) * 1000;
  timestamp_ = edm::Timestamp((seconds << 32) | microseconds);
  highVoltageReady_ = dcs.highVoltageReady;
  highVoltageValid_ = dcs.highVoltageValid;
  magnetCurrent_ = dcs.magnetCurrent;
}

DCSRecord::~DCSRecord() {}

std::ostream& operator<<(std::ostream& s, const DCSRecord& dcs) {
  const time_t ts = dcs.timestamp().unixTime();

  s << "timeStamp:            " << asctime(localtime(&ts));

  std::streamsize ss = s.precision();
  s.setf(std::ios::fixed);
  s.precision(3);
  s << "Magnet current (A):   " << std::fixed << std::setprecision(3) << dcs.magnetCurrent() << std::endl;
  s.unsetf(std::ios::fixed);
  s.precision(ss);

  s << "HV state per partition:" << std::endl;

  for (unsigned int i = 0; i < DCSRecord::Partition::Last; ++i) {
    s << "   " << std::setw(7) << std::left << dcs.partitionName(i) << ": "
      << (!dcs.highVoltageValid(i) ? "N/A" : (dcs.highVoltageReady(i) ? "READY" : "OFF")) << std::endl;
  }

  return s;
}

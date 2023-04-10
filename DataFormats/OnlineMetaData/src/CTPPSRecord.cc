#include <iomanip>
#include <ctime>

#include "DataFormats/OnlineMetaData/interface/CTPPSRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"

const CTPPSRecord::RomanPotNames CTPPSRecord::romanPotNames_ = {{"RP_45_210_FR_BT",
                                                                 "RP_45_210_FR_HR",
                                                                 "RP_45_210_FR_TP",
                                                                 "RP_45_220_C1",
                                                                 "RP_45_220_FR_BT",
                                                                 "RP_45_220_FR_HR",
                                                                 "RP_45_220_FR_TP",
                                                                 "RP_45_220_NR_BT",
                                                                 "RP_45_220_NR_HR",
                                                                 "RP_45_220_NR_TP",
                                                                 "RP_56_210_FR_BT",
                                                                 "RP_56_210_FR_HR",
                                                                 "RP_56_210_FR_TP",
                                                                 "RP_56_220_C1",
                                                                 "RP_56_220_FR_BT",
                                                                 "RP_56_220_FR_HR",
                                                                 "RP_56_220_FR_TP",
                                                                 "RP_56_220_NR_BT",
                                                                 "RP_56_220_NR_HR",
                                                                 "RP_56_220_NR_TP"}};

const std::array<std::string, 4> CTPPSRecord::statusNames_ = {{"unused", "bad", "warning", "ok"}};

CTPPSRecord::CTPPSRecord() : timestamp_(edm::Timestamp::invalidTimestamp()), status_(0) {}

CTPPSRecord::CTPPSRecord(const online::CTPPS_v1& ctpps) {
  // DIP timestamp is in milliseconds
  const uint64_t seconds = ctpps.timestamp / 1000;
  const uint32_t microseconds = (ctpps.timestamp % 1000) * 1000;
  timestamp_ = edm::Timestamp((seconds << 32) | microseconds);
  status_ = ctpps.status;
}

CTPPSRecord::~CTPPSRecord() {}

std::ostream& operator<<(std::ostream& s, const CTPPSRecord& ctpps) {
  const time_t ts = ctpps.timestamp().unixTime();

  s << "timeStamp:            " << asctime(localtime(&ts));
  s << "Roman pot states:" << std::endl;

  for (uint8_t i = 0; i < CTPPSRecord::RomanPot::Last; ++i) {
    s << "   " << std::setw(16) << std::left << ctpps.romanPotName(i) << ": " << ctpps.statusName(i) << std::endl;
  }

  return s;
}

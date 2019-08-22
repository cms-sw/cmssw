#include <iomanip>
#include <ctime>

#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"

OnlineLuminosityRecord::OnlineLuminosityRecord()
    : timestamp_(edm::Timestamp::invalidTimestamp()), instLumi_(0), avgPileUp_(0), lumiSection_(0), lumiNibble_(0) {}

OnlineLuminosityRecord::OnlineLuminosityRecord(const online::Luminosity_v1& lumi) {
  // DIP timestamp is in milliseconds
  const uint64_t seconds = lumi.timestamp / 1000;
  const uint32_t microseconds = (lumi.timestamp % 1000) * 1000;
  timestamp_ = edm::Timestamp((seconds << 32) | microseconds);
  instLumi_ = lumi.instLumi;
  avgPileUp_ = lumi.avgPileUp;
  lumiSection_ = lumi.lumiSection;
  lumiNibble_ = lumi.lumiNibble;
}

OnlineLuminosityRecord::~OnlineLuminosityRecord() {}

std::ostream& operator<<(std::ostream& s, const OnlineLuminosityRecord& luminosity) {
  const time_t ts = luminosity.timestamp().unixTime();

  s << "timeStamp:        " << asctime(localtime(&ts));
  s << "lumiSection:      " << luminosity.lumiSection() << std::endl;
  s << "lumiNibble:       " << luminosity.lumiNibble() << std::endl;

  std::streamsize ss = s.precision();
  s.setf(std::ios::fixed);
  s.precision(2);
  s << "instLumi:         " << luminosity.instLumi() << std::endl;
  s << "avgPileUp:        " << luminosity.avgPileUp() << std::endl;
  s.unsetf(std::ios::fixed);
  s.precision(ss);

  return s;
}

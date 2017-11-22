#include <iomanip>
#include <time.h>

#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"


OnlineLuminosityRecord::OnlineLuminosityRecord() :
  timestamp_(edm::Timestamp::invalidTimestamp()),
  lumiSection_(0),
  lumiNibble_(0),
  instLumi_(0),
  avgPileUp_(0)
{}


OnlineLuminosityRecord::OnlineLuminosityRecord(const onlineMetaData::Luminosity_v1& lumi)
{
  // DIP timestamp is in milliseconds
  const uint64_t seconds = lumi.timestamp / 1000;
  const uint32_t microseconds = (lumi.timestamp % 1000) * 1000;
  timestamp_ = edm::Timestamp((seconds<<32) | microseconds );
  lumiSection_ = lumi.lumiSection;
  lumiNibble_ = lumi.lumiNibble;
  instLumi_ = lumi.instLumi;
  avgPileUp_ = lumi.avgPileUp;
}


OnlineLuminosityRecord::~OnlineLuminosityRecord() {}


std::ostream& operator<<(std::ostream& s, const OnlineLuminosityRecord& luminosity)
{
  const time_t ts = luminosity.getTimestamp().unixTime();

  s << "timeStamp:        " << asctime(localtime(&ts));
  s << "lumiSection:      " << luminosity.getLumiSection() << std::endl;
  s << "lumiNibble:       " << luminosity.getLumiNibble() << std::endl;

  std::streamsize ss = s.precision();
  s.setf(std::ios::fixed);
  s.precision(2);
  s << "instLumi:         " << luminosity.getInstLumi() << std::endl;
  s << "avgPileUp:        " << luminosity.getAvgPileUp() << std::endl;
  s.unsetf(std::ios::fixed);
  s.precision(ss);

  return s;
}

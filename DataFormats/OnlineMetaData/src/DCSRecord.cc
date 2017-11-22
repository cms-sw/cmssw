#include <iomanip>

#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"


DCSRecord::DCSRecord() :
  timestamp_(edm::Timestamp::invalidTimestamp()),
  highVoltageReady_(0),
  magnetCurrent_(-1),
  magneticField_(-1)
{}


DCSRecord::DCSRecord(const onlineMetaData::DCS_v1& dcs)
{
  // DIP timestamp is in milliseconds
  const uint64_t seconds = dcs.timestamp / 1000;
  const uint32_t microseconds = (dcs.timestamp % 1000) * 1000;
  timestamp_ = edm::Timestamp((seconds<<32) | microseconds );
  highVoltageReady_ = dcs.highVoltageReady;
  magnetCurrent_ = dcs.magnetCurrent;
  magneticField_ = dcs.magneticField;
}


DCSRecord::~DCSRecord() {}


std::ostream& operator<<(std::ostream& s, const DCSRecord& dcs)
{
  const time_t ts = dcs.getTimestamp().unixTime();

  s << "timeStamp:            " << asctime(localtime(&ts));

  std::streamsize ss = s.precision();
  s.setf(std::ios::fixed);
  s.precision(3);
  s << "Magnet current (A):   " << std::fixed << std::setprecision(3) << dcs.getMagnetCurrent() << std::endl;
  s << "Magnetic field (T):   " << std::fixed << std::setprecision(3) << dcs.getMagneticField() << std::endl;
  s.unsetf(std::ios::fixed);
  s.precision(ss);

  s << "HV state per partition:" << std::endl;

  for (unsigned int i = 0; i < DCSRecord::Partition::Last; ++i) {
    s << "   " << std::setw(7) << std::left << dcs.getPartitionName(i)
      << ": " << (dcs.highVoltageReady(i)?"READY":"OFF")
      << std::endl;
  }

  return s;
}

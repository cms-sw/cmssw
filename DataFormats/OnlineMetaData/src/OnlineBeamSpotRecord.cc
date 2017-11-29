#include <iomanip>
#include <ctime>

#include "DataFormats/OnlineMetaData/interface/OnlineBeamSpotRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"


OnlineBeamSpotRecord::OnlineBeamSpotRecord() :
  timestamp_(edm::Timestamp::invalidTimestamp()),
  x_(0),
  y_(0),
  z_(0),
  dxdz_(0),
  dydz_(0),
  errX_(0),
  errY_(0),
  errZ_(0),
  errDxdz_(0),
  errDydz_(0),
  widthX_(0),
  widthY_(0),
  sigmaZ_(0),
  errWidthX_(0),
  errWidthY_(0),
  errSigmaZ_(0)
{}


OnlineBeamSpotRecord::OnlineBeamSpotRecord(const onlineMetaData::BeamSpot_v1& beamSpot)
{
  // DIP timestamp is in milliseconds
  const uint64_t seconds = beamSpot.timestamp / 1000;
  const uint32_t microseconds = (beamSpot.timestamp % 1000) * 1000;
  timestamp_ = edm::Timestamp((seconds<<32) | microseconds );
  x_ = beamSpot.x;
  y_ = beamSpot.y;
  z_ = beamSpot.z;
  dxdz_ = beamSpot.dxdz;
  dydz_ = beamSpot.dydz;
  errX_ = beamSpot.errX;
  errY_ = beamSpot.errY;
  errZ_ = beamSpot.errZ;
  errDxdz_ = beamSpot.errDxdz;
  errDydz_ = beamSpot.errDydz;
  widthX_ = beamSpot.widthX;
  widthY_ = beamSpot.widthY;
  sigmaZ_ = beamSpot.sigmaZ;
  errWidthX_ = beamSpot.errWidthX;
  errWidthY_ = beamSpot.errWidthY;
  errSigmaZ_ = beamSpot.errSigmaZ;
}


OnlineBeamSpotRecord::~OnlineBeamSpotRecord() {}


std::ostream& operator<<(std::ostream& s, const OnlineBeamSpotRecord& beamSpot)
{
  const time_t ts = beamSpot.timestamp().unixTime();

  s << "timeStamp:         " << asctime(localtime(&ts));

  std::streamsize ss = s.precision();
  s.setf(std::ios::fixed);
  s.precision(6);
  s << "x:                 " << beamSpot.x() << std::endl;
  s << "y:                 " << beamSpot.y() << std::endl;
  s << "z:                 " << beamSpot.z() << std::endl;
  s << "dxdz:              " << beamSpot.dxdz() << std::endl;
  s << "dydz:              " << beamSpot.dydz() << std::endl;
  s << "err of x:          " << beamSpot.errX() << std::endl;
  s << "err of y:          " << beamSpot.errX() << std::endl;
  s << "err of z:          " << beamSpot.errZ() << std::endl;
  s << "err of dxdz:       " << beamSpot.errDxdz() << std::endl;
  s << "err of dydz:       " << beamSpot.errDydz() << std::endl;
  s << "width in x:        " << beamSpot.widthX() << std::endl;
  s << "width in y:        " << beamSpot.widthY() << std::endl;
  s << "sigma z:           " << beamSpot.sigmaZ() << std::endl;
  s << "err of width in x: " << beamSpot.errWidthX() << std::endl;
  s << "err of width in y  " << beamSpot.errWidthY() << std::endl;
  s << "err of sigma z:    " << beamSpot.errSigmaZ() << std::endl;
  s.unsetf(std::ios::fixed);
  s.precision(ss);

  return s;
}

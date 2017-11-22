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
  const time_t ts = beamSpot.getTimestamp().unixTime();

  s << "timeStamp:         " << asctime(localtime(&ts));

  std::streamsize ss = s.precision();
  s.setf(std::ios::fixed);
  s.precision(6);
  s << "x:                 " << beamSpot.getX() << std::endl;
  s << "y:                 " << beamSpot.getY() << std::endl;
  s << "z:                 " << beamSpot.getZ() << std::endl;
  s << "dxdz:              " << beamSpot.getDxdz() << std::endl;
  s << "dydz:	           " << beamSpot.getDydz() << std::endl;
  s << "err of x:          " << beamSpot.getErrX() << std::endl;
  s << "err of y:          " << beamSpot.getErrX() << std::endl;
  s << "err of z:          " << beamSpot.getErrZ() << std::endl;
  s << "err of dxdz:       " << beamSpot.getErrDxdz() << std::endl;
  s << "err of dydz:       " << beamSpot.getErrDydz() << std::endl;
  s << "width in x:        " << beamSpot.getWidthX() << std::endl;
  s << "width in y:        " << beamSpot.getWidthY() << std::endl;
  s << "sigma z:           " << beamSpot.getSigmaZ() << std::endl;
  s << "err of width in x: " << beamSpot.getErrWidthX() << std::endl;
  s << "err of width in y  " << beamSpot.getErrWidthY() << std::endl;
  s << "err of sigma z:    " << beamSpot.getErrSigmaZ() << std::endl;
  s.unsetf(std::ios::fixed);
  s.precision(ss);

  return s;
}

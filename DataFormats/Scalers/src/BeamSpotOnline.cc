/*
 *   File: DataFormats/Scalers/src/BeamSpotOnline.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"
#include <cstdio>
#include <ostream>

BeamSpotOnline::BeamSpotOnline()
    : trigType_(0),
      eventID_(0),
      sourceID_(0),
      bunchNumber_(0),
      version_(0),
      collectionTime_(0, 0),
      x_((float)0.0),
      y_((float)0.0),
      z_((float)0.0),
      dxdz_((float)0.0),
      dydz_((float)0.0),
      err_x_((float)0.0),
      err_y_((float)0.0),
      err_z_((float)0.0),
      err_dxdz_((float)0.0),
      err_dydz_((float)0.0),
      width_x_((float)0.0),
      width_y_((float)0.0),
      sigma_z_((float)0.0),
      err_width_x_((float)0.0),
      err_width_y_((float)0.0),
      err_sigma_z_((float)0.0) {}

BeamSpotOnline::BeamSpotOnline(const unsigned char* rawData) {
  BeamSpotOnline();

  struct ScalersEventRecordRaw_v4 const* raw = reinterpret_cast<struct ScalersEventRecordRaw_v4 const*>(rawData);
  trigType_ = (raw->header >> 56) & 0xFULL;
  eventID_ = (raw->header >> 32) & 0x00FFFFFFULL;
  sourceID_ = (raw->header >> 8) & 0x00000FFFULL;
  bunchNumber_ = (raw->header >> 20) & 0xFFFULL;

  version_ = raw->version;
  if (version_ >= 4) {
    collectionTime_.set_tv_sec(static_cast<long>(raw->beamSpotOnline.collectionTime_sec));
    collectionTime_.set_tv_nsec(raw->beamSpotOnline.collectionTime_nsec);
    x_ = raw->beamSpotOnline.x;
    y_ = raw->beamSpotOnline.y;
    z_ = raw->beamSpotOnline.z;
    dxdz_ = raw->beamSpotOnline.dxdz;
    dydz_ = raw->beamSpotOnline.dydz;
    err_x_ = raw->beamSpotOnline.err_x;
    err_y_ = raw->beamSpotOnline.err_y;
    err_z_ = raw->beamSpotOnline.err_z;
    err_dxdz_ = raw->beamSpotOnline.err_dxdz;
    err_dydz_ = raw->beamSpotOnline.err_dydz;
    width_x_ = raw->beamSpotOnline.width_x;
    width_y_ = raw->beamSpotOnline.width_y;
    sigma_z_ = raw->beamSpotOnline.sigma_z;
    err_width_x_ = raw->beamSpotOnline.err_width_x;
    err_width_y_ = raw->beamSpotOnline.err_width_y;
    err_sigma_z_ = raw->beamSpotOnline.err_sigma_z;
  }
}

BeamSpotOnline::~BeamSpotOnline() {}

/// Pretty-print operator for BeamSpotOnline
std::ostream& operator<<(std::ostream& s, const BeamSpotOnline& c) {
  char zeit[128];
  constexpr size_t kLineBufferSize = 157;
  char line[kLineBufferSize];
  struct tm* hora;

  s << "BeamSpotOnline    Version: " << c.version() << "   SourceID: " << c.sourceID() << std::endl;

  timespec ts = c.collectionTime();
  hora = gmtime(&ts.tv_sec);
  strftime(zeit, sizeof(zeit), "%Y.%m.%d %H:%M:%S", hora);
  snprintf(line, kLineBufferSize, " CollectionTime: %s.%9.9d", zeit, (int)ts.tv_nsec);
  s << line << std::endl;

  snprintf(line,
           kLineBufferSize,
           " TrigType: %d   EventID: %d    BunchNumber: %d",
           c.trigType(),
           c.eventID(),
           c.bunchNumber());
  s << line << std::endl;

  snprintf(
      line, kLineBufferSize, "    x: %e +/- %e   width: %e +/- %e", c.x(), c.err_x(), c.width_x(), c.err_width_x());
  s << line << std::endl;

  snprintf(
      line, kLineBufferSize, "    y: %e +/- %e   width: %e +/- %e", c.y(), c.err_y(), c.width_y(), c.err_width_y());
  s << line << std::endl;

  snprintf(
      line, kLineBufferSize, "    z: %e +/- %e   sigma: %e +/- %e", c.z(), c.err_z(), c.sigma_z(), c.err_sigma_z());
  s << line << std::endl;

  snprintf(
      line, kLineBufferSize, " dxdy: %e +/- %e    dydz: %e +/- %e", c.dxdz(), c.err_dxdz(), c.dydz(), c.err_dydz());
  s << line << std::endl;
  return s;
}

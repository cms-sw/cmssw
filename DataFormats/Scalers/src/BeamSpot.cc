/*
 *   File: DataFormats/Scalers/src/BeamSpot.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/BeamSpot.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"
#include <cstdio>
#include <ostream>

BeamSpot::BeamSpot() : 
   trigType_(0),
   eventID_(0),
   sourceID_(0),
   bunchNumber_(0),
   version_(0),
   collectionTime_(0,0),
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
   err_sigma_z_((float)0.0)
{ 
}

BeamSpot::BeamSpot(const unsigned char * rawData)
{ 
  BeamSpot();

  struct ScalersEventRecordRaw_v4 * raw 
    = (struct ScalersEventRecordRaw_v4 *)rawData;
  trigType_     = ( raw->header >> 56 ) &        0xFULL;
  eventID_      = ( raw->header >> 32 ) & 0x00FFFFFFULL;
  sourceID_     = ( raw->header >>  8 ) & 0x00000FFFULL;
  bunchNumber_  = ( raw->header >> 20 ) &      0xFFFULL;

  version_ = raw->version;
  if ( version_ >= 4 )
  {
    collectionTime_.set_tv_sec(static_cast<long>(raw->beamSpot.collectionTime_sec));
    collectionTime_.set_tv_nsec(raw->beamSpot.collectionTime_nsec);
    x_           = raw->beamSpot.x;
    y_           = raw->beamSpot.y;
    z_           = raw->beamSpot.z;
    dxdz_        = raw->beamSpot.dxdz;
    dydz_        = raw->beamSpot.dydz;
    err_x_       = raw->beamSpot.err_x;
    err_y_       = raw->beamSpot.err_y;
    err_z_       = raw->beamSpot.err_z;
    err_dxdz_    = raw->beamSpot.err_dxdz;
    err_dydz_    = raw->beamSpot.err_dydz;
    width_x_     = raw->beamSpot.width_x;
    width_y_     = raw->beamSpot.width_y;
    sigma_z_     = raw->beamSpot.sigma_z;
    err_width_x_ = raw->beamSpot.err_width_x;
    err_width_y_ = raw->beamSpot.err_width_y;
    err_sigma_z_ = raw->beamSpot.err_sigma_z;
  }
}

BeamSpot::~BeamSpot() { } 


/// Pretty-print operator for BeamSpot
std::ostream& operator<<(std::ostream& s, const BeamSpot& c) 
{
  char zeit[128];
  char line[128];
  struct tm * hora;

  s << "BeamSpot    Version: " << c.version() << 
    "   SourceID: "<< c.sourceID() << std::endl;

  timespec ts = c.collectionTime();
  hora = gmtime(&ts.tv_sec);
  strftime(zeit, sizeof(zeit), "%Y.%m.%d %H:%M:%S", hora);
  sprintf(line, " CollectionTime: %s.%9.9d", zeit, 
	  (int)ts.tv_nsec);
  s << line << std::endl;

  sprintf(line, " TrigType: %d   EventID: %d    BunchNumber: %d", 
	  c.trigType(), c.eventID(), c.bunchNumber());
  s << line << std::endl;

  sprintf(line,"       x: %e +/- %e   y: %e +/- %e    z: %e +/- %e",
	  c.x(), c.err_x(), c.y(), c.err_y(), c.z(), c.err_z());
  s << line << std::endl;

  sprintf(line," width_x: %e +/- %e    width_y: %e +/- %e   sigma_z: %e +/- %e",
	  c.width_x(), c.err_width_x(), 
	  c.width_y(), c.err_width_y(), 
	  c.sigma_z(), c.err_sigma_z());
  s << line << std::endl;

  sprintf(line," dxdy: %e +/- %e    dxdz: %e +/- %e",
	  c.dxdz(), c.err_dxdz(), c.dydz(), c.err_dydz());
  s << line << std::endl;
  return s;
}

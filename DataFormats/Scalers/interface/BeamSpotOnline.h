/*
 *  File: DataFormats/Scalers/interface/BeamSpotOnline.h   (W.Badgett)
 *
 *  The online computed BeamSpotOnline value
 *
 */

#ifndef DATAFORMATS_SCALERS_BEAMSPOT_H
#define DATAFORMATS_SCALERS_BEAMSPOT_H

#include "DataFormats/Scalers/interface/TimeSpec.h"

#include <ctime>
#include <iosfwd>
#include <vector>
#include <string>

/*! \file BeamSpotOnline.h
 * \Header file for online BeamSpotOnline value
 * 
 * \author: William Badgett
 *
 */

/// \class BeamSpotOnline.h
/// \brief Persistable copy of online BeamSpotOnline value

class BeamSpotOnline {
public:
  BeamSpotOnline();
  BeamSpotOnline(const unsigned char* rawData);
  virtual ~BeamSpotOnline();

  /// name method
  std::string name() const { return "BeamSpotOnline"; }

  /// empty method (= false)
  bool empty() const { return false; }

  unsigned int trigType() const { return (trigType_); }
  unsigned int eventID() const { return (eventID_); }
  unsigned int sourceID() const { return (sourceID_); }
  unsigned int bunchNumber() const { return (bunchNumber_); }

  int version() const { return (version_); }
  timespec collectionTime() const { return (collectionTime_.get_timespec()); }

  float x() const { return (x_); }
  float y() const { return (y_); }
  float z() const { return (z_); }
  float dxdz() const { return (dxdz_); }
  float dydz() const { return (dydz_); }
  float err_x() const { return (err_x_); }
  float err_y() const { return (err_y_); }
  float err_z() const { return (err_z_); }
  float err_dxdz() const { return (err_dxdz_); }
  float err_dydz() const { return (err_dydz_); }
  float width_x() const { return (width_x_); }
  float width_y() const { return (width_y_); }
  float sigma_z() const { return (sigma_z_); }
  float err_width_x() const { return (err_width_x_); }
  float err_width_y() const { return (err_width_y_); }
  float err_sigma_z() const { return (err_sigma_z_); }

  /// equality operator
  int operator==(const BeamSpotOnline& e) const { return false; }

  /// inequality operator
  int operator!=(const BeamSpotOnline& e) const { return false; }

protected:
  unsigned int trigType_;
  unsigned int eventID_;
  unsigned int sourceID_;
  unsigned int bunchNumber_;

  int version_;

  TimeSpec collectionTime_;
  float x_;
  float y_;
  float z_;
  float dxdz_;
  float dydz_;
  float err_x_;
  float err_y_;
  float err_z_;
  float err_dxdz_;
  float err_dydz_;
  float width_x_;
  float width_y_;
  float sigma_z_;
  float err_width_x_;
  float err_width_y_;
  float err_sigma_z_;
};

/// Pretty-print operator for BeamSpotOnline
std::ostream& operator<<(std::ostream& s, const BeamSpotOnline& c);

typedef std::vector<BeamSpotOnline> BeamSpotOnlineCollection;

#endif

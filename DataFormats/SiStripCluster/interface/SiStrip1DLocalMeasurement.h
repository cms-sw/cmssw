#ifndef DATAFORMATS_SISTRIP1DLOCALMEASUREMENT_H
#define DATAFORMATS_SISTRIP1DLOCALMEASUREMENT_H

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

class SiStrip1DLocalMeasurement {
public:

  SiStrip1DLocalMeasurement() :
    detId_(0), position_(0.), error_(0.) {}

  SiStrip1DLocalMeasurement(cms::DetId detid, const float position, const float error) :
    detId_(detid), position_(position), error_(error) {}

  cms::DetId geographicalId() const {return detId_;}

  float position() const {return position_;}
  float error() const {return error_;}

private:

  cms::DetId           detId_;

  float                position_;
  float                error_;

};

// Comparison operators
inline bool operator<( const SiStrip1DLocalMeasurement& one, const SiStrip1DLocalMeasurement& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else if ( one.geographicalId().rawId() > other.geographicalId().rawId() ) {
    return false;
  } else {
    if ( one.position() <= other.position() ) {
    return true;
  } else {
    return false;
    }
  }
}

#endif // DATAFORMATS_SISTRIP1DLOCALMEASUREMENT_H

#ifndef DATAFORMATS_HCALRECHIT_HcalCalibRECHIT_H
#define DATAFORMATS_HCALRECHIT_HcalCalibRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"

/** \class HcalCalibRecHit
 *  
 * $Date: 2006/06/27 15:49:21 $
 * $Revision: 1.1 $
 *\author J. Mans - Minnesota
 */
class HcalCalibRecHit {
public:
  typedef HcalCalibDetId key_type;

  HcalCalibRecHit();
  HcalCalibRecHit(const HcalCalibDetId& id, float amplitude, float time);
  /// get the amplitude (generally fC, but can vary)
  float amplitude() const { return amplitude_; }
  /// get the hit time (if available)
  float time() const { return time_; }
  /// get the id
  HcalCalibDetId id() const { return id_; }
private:
  HcalCalibDetId id_;
  float amplitude_,time_;
};

std::ostream& operator<<(std::ostream& s, const HcalCalibRecHit& hit);

#endif

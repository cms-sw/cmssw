#include "DataFormats/HcalRecHit/interface/HcalCalibRecHit.h"

HcalCalibRecHit::HcalCalibRecHit() : id_(), amplitude_(0), time_(0) {}

HcalCalibRecHit::HcalCalibRecHit(const HcalCalibDetId& id, float ampl, float time)
    : id_(id), amplitude_(ampl), time_(time) {}

std::ostream& operator<<(std::ostream& s, const HcalCalibRecHit& hit) {
  return s << hit.id() << ": " << hit.amplitude() << " , " << hit.time() << " ns";
}

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include <math.h>

EcalUncalibratedRecHit::EcalUncalibratedRecHit() :
     amplitude_(0.), pedestal_(0.), jitter_(0.), chi2_(10000.), flags_(0) { }

EcalUncalibratedRecHit::EcalUncalibratedRecHit(const DetId& id, const double& ampl, const double& ped,
                          const double& jit, const double& chi2, const uint32_t &flags) :
     amplitude_(ampl), pedestal_(ped), jitter_(jit), chi2_(chi2), flags_(flags), id_(id) { }

EcalUncalibratedRecHit::~EcalUncalibratedRecHit() {
}

bool EcalUncalibratedRecHit::isSaturated() const {
  return ( flags_ == kSaturated );
}

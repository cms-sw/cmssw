#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

EcalUncalibratedRecHit::EcalUncalibratedRecHit() :
     amplitude_(0.), pedestal_(0.), jitter_(0.), chi2_(10000.) { }

EcalUncalibratedRecHit::EcalUncalibratedRecHit(const DetId& id, const double& ampl, const double& ped,
                          const double& jit, const double& chi2) :
     amplitude_(ampl), pedestal_(ped), jitter_(jit), chi2_(chi2), id_(id) { }

EcalUncalibratedRecHit::~EcalUncalibratedRecHit() {
}

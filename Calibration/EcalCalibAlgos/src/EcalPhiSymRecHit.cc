#include "Calibration/EcalCalibAlgos/interface/EcalPhiSymRecHit.h"

//**********constructors******************************************************************
EcalPhiSymRecHit::EcalPhiSymRecHit()
    : id_(0), eeRing_(0), chStatus_(0), nHits_(0), etSum_(1, 0.), et2Sum_(0), lcSum_(0), lc2Sum_(0) {}

EcalPhiSymRecHit::EcalPhiSymRecHit(uint32_t id, unsigned int nMisCalibV, unsigned int status)
    : id_(id), eeRing_(0), chStatus_(status), nHits_(0), etSum_(nMisCalibV, 0.), et2Sum_(0), lcSum_(0), lc2Sum_(0) {}

EcalPhiSymRecHit::EcalPhiSymRecHit(uint32_t id, std::vector<float>& etValues, unsigned int status)
    : id_(id),
      eeRing_(0),
      chStatus_(status),
      nHits_(0),
      etSum_(etValues.begin(), etValues.end()),
      et2Sum_(0),
      lcSum_(0),
      lc2Sum_(0) {}

//**********utils*************************************************************************
void EcalPhiSymRecHit::addHit(const std::vector<float>& etValues, const float laserCorr) {
  if (etValues[0] > 0.) {
    ++nHits_;
    et2Sum_ += etValues[0] * etValues[0];
    lcSum_ += laserCorr;
    lc2Sum_ += laserCorr * laserCorr;
  }
  for (unsigned int i = 0; i < std::min(etSum_.size(), etValues.size()); ++i)
    etSum_[i] += etValues[i];
}

void EcalPhiSymRecHit::reset() {
  nHits_ = 0.;
  et2Sum_ = 0.;
  lcSum_ = 0.;
  lc2Sum_ = 0.;
  etSum_ = std::vector<float>(etSum_.size(), 0.);
}

//**********operators*********************************************************************

EcalPhiSymRecHit& EcalPhiSymRecHit::operator+=(const EcalPhiSymRecHit& rhs) {
  // assume same id, do not check channel status
  assert("EcalPhiSymRecHit operator+= : attempting to sum RecHits belonging to different channels" &&
         (id_ == rhs.rawId()));
  nHits_ += rhs.nHits();
  et2Sum_ += rhs.sumEt2();
  lcSum_ += rhs.lcSum();
  lc2Sum_ += rhs.lc2Sum();
  for (unsigned int i = 0; i < etSum_.size(); ++i)
    etSum_[i] += rhs.sumEt(i);

  return *this;
}

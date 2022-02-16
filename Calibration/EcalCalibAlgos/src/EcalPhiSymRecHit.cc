#include "Calibration/EcalCalibAlgos/interface/EcalPhiSymRecHit.h"

//**********constructors******************************************************************
EcalPhiSymRecHit::EcalPhiSymRecHit()
    : id_(0), chStatus_(0), nHits_(0), etSum_(1, 0.), et2Sum_(0), lcSum_(0), lc2Sum_(0) {}

EcalPhiSymRecHit::EcalPhiSymRecHit(uint32_t id, unsigned int nMisCalibV, unsigned int status)
    : id_(id), chStatus_(status), nHits_(0), etSum_(nMisCalibV, 0.), et2Sum_(0), lcSum_(0), lc2Sum_(0) {}

EcalPhiSymRecHit::EcalPhiSymRecHit(uint32_t id, std::vector<float>& etValues, unsigned int status)
    : id_(id),
      chStatus_(status),
      nHits_(0),
      etSum_(etValues.begin(), etValues.end()),
      et2Sum_(0),
      lcSum_(0),
      lc2Sum_(0) {}

//**********destructor********************************************************************
EcalPhiSymRecHit::~EcalPhiSymRecHit() {}

//**********utils*************************************************************************

void EcalPhiSymRecHit::AddHit(float* etValues, float laserCorr) {
  if (etValues[0] > 0.) {
    ++nHits_;
    etSum_[0] += etValues[0];
    et2Sum_ += etValues[0] * etValues[0];
    lcSum_ += laserCorr;
    lc2Sum_ += laserCorr * laserCorr;
  }
  for (unsigned int i = 0; i < etSum_.size(); ++i)
    etSum_[i] += etValues[i];
}

void EcalPhiSymRecHit::AddHit(std::vector<float>& etValues, float laserCorr) { AddHit(etValues.data()); }

void EcalPhiSymRecHit::Reset() {
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
         (id_ == rhs.GetRawId()));
  nHits_ += rhs.GetNhits();
  et2Sum_ += rhs.GetSumEt2();
  lcSum_ += rhs.GetLCSum();
  lc2Sum_ += rhs.GetLC2Sum();
  for (unsigned int i = 0; i < etSum_.size(); ++i)
    etSum_[i] += rhs.GetSumEt(i);

  return *this;
}

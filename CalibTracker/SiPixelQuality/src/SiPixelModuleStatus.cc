#include <iostream>
#include <map>
#include <cmath>
#include <vector>

#include <TMath.h>
#include <TH1.h>

#include "CalibTracker/SiPixelQuality/interface/SiPixelModuleStatus.h"

// ----------------------------------------------------------------------
SiPixelModuleStatus::SiPixelModuleStatus(int detId, int nrocs) : fDetid_(detId), fNrocs_(nrocs) {
  for (int i = 0; i < fNrocs_; ++i) {
    SiPixelRocStatus a;
    fRocs_.push_back(a);
  }
};

// ----------------------------------------------------------------------
SiPixelModuleStatus::~SiPixelModuleStatus(){};

// ----------------------------------------------------------------------
void SiPixelModuleStatus::fillDIGI(int iroc) {
  if (iroc < fNrocs_)
    fRocs_[iroc].fillDIGI();
}
// ----------------------------------------------------------------------
void SiPixelModuleStatus::fillFEDerror25(PixelFEDChannel ch) {
  int roc_first = int(ch.roc_first);
  int roc_last = int(ch.roc_last);
  for (int iroc = 0; iroc < fNrocs_; iroc++) {
    if (iroc >= roc_first && iroc <= roc_last) {
      fRocs_[iroc].fillFEDerror25();
    }
  }
}
// ----------------------------------------------------------------------
int SiPixelModuleStatus::detid() { return fDetid_; }
// ----------------------------------------------------------------------
int SiPixelModuleStatus::nrocs() { return fNrocs_; }
// ----------------------------------------------------------------------
void SiPixelModuleStatus::setDetId(int detid) { fDetid_ = detid; }
// ----------------------------------------------------------------------
void SiPixelModuleStatus::setNrocs(int nRoc) { fNrocs_ = nRoc; }

// ----------------------------------------------------------------------
void SiPixelModuleStatus::updateDIGI(int iroc, unsigned int nhit) {
  if (iroc < fNrocs_)
    fRocs_[iroc].updateDIGI(nhit);
}
// ----------------------------------------------------------------------
void SiPixelModuleStatus::updateFEDerror25(int iroc, bool fedError25) {
  if (iroc < fNrocs_)
    fRocs_[iroc].updateFEDerror25(fedError25);
}

// ----------------------------------------------------------------------
unsigned int SiPixelModuleStatus::digiOccROC(int iroc) { return (iroc < fNrocs_ ? fRocs_[iroc].digiOccROC() : 0); }
// ----------------------------------------------------------------------
bool SiPixelModuleStatus::fedError25(int iroc) { return (iroc < fNrocs_ ? fRocs_[iroc].isFEDerror25() : false); }
// ----------------------------------------------------------------------
unsigned int SiPixelModuleStatus::digiOccMOD() {
  unsigned int count(0);
  for (int iroc = 0; iroc < fNrocs_; ++iroc) {
    count += digiOccROC(iroc);
  }
  return count;
}

// ----------------------------------------------------------------------
double SiPixelModuleStatus::perRocDigiOcc() {
  unsigned int ave(0);
  for (int iroc = 0; iroc < fNrocs_; ++iroc) {
    unsigned int inc = digiOccROC(iroc);
    ave += inc;
  }
  return (1.0 * ave) / fNrocs_;
}

double SiPixelModuleStatus::perRocDigiOccVar() {
  double fModAverage = SiPixelModuleStatus::perRocDigiOcc();

  double sig = 1.0;
  for (int iroc = 0; iroc < fNrocs_; ++iroc) {
    unsigned int inc = digiOccROC(iroc);
    sig += (fModAverage - inc) * (fModAverage - inc);
  }

  double fModSigma = sig / (fNrocs_ - 1);
  return TMath::Sqrt(fModSigma);
}

// ----------------------------------------------------------------------
// Return the address not the value of ROC status
SiPixelRocStatus* SiPixelModuleStatus::getRoc(int iroc) { return (iroc < fNrocs_ ? &fRocs_[iroc] : nullptr); }

// ----------------------------------------------------------------------
void SiPixelModuleStatus::updateModuleDIGI(int iroc, unsigned int nhits) {
  if (iroc < fNrocs_)
    fRocs_[iroc].updateDIGI(nhits);
}
// ----------------------------------------------------------------------
void SiPixelModuleStatus::updateModuleStatus(SiPixelModuleStatus newData) {
  bool isSameModule = true;
  if (fDetid_ != newData.detid() || fNrocs_ != newData.nrocs()) {
    isSameModule = false;
  }

  if (isSameModule) {
    for (int iroc = 0; iroc < fNrocs_; ++iroc) {  // loop over rocs
      //update occupancy
      fRocs_[iroc].updateDIGI(newData.digiOccROC(iroc));
      //update FEDerror25
      fRocs_[iroc].updateFEDerror25(newData.fedError25(iroc));

    }  // loop over rocs

  }  // if same module
}

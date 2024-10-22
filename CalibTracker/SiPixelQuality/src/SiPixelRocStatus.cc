#include <iostream>
#include <map>
#include <cmath>
#include <vector>

#include "CalibTracker/SiPixelQuality/interface/SiPixelRocStatus.h"

using namespace std;

// ----------------------------------------------------------------------
SiPixelRocStatus::SiPixelRocStatus() {
  fDC_ = 0;
  isFEDerror25_ = false;
}

// ----------------------------------------------------------------------
SiPixelRocStatus::~SiPixelRocStatus() {}

// ----------------------------------------------------------------------
void SiPixelRocStatus::fillDIGI() { fDC_++; }
// ----------------------------------------------------------------------
void SiPixelRocStatus::fillFEDerror25() { isFEDerror25_ = true; }

// ----------------------------------------------------------------------
void SiPixelRocStatus::updateDIGI(unsigned int hits) { fDC_ += hits; }
// ----------------------------------------------------------------------
/*AND logic to update FEDerror25*/
void SiPixelRocStatus::updateFEDerror25(bool fedError25) { isFEDerror25_ = isFEDerror25_ && fedError25; }

// ----------------------------------------------------------------------
unsigned int SiPixelRocStatus::digiOccROC() { return fDC_; }
// ----------------------------------------------------------------------
bool SiPixelRocStatus::isFEDerror25() { return isFEDerror25_; }

#include <iostream>
#include <map>
#include <cmath>
#include <vector>

#include "CalibTracker/SiPixelQuality/interface/SiPixelRocStatus.h"

using namespace std;

// ----------------------------------------------------------------------
SiPixelRocStatus::SiPixelRocStatus() {
  fDC = 0;
  isStuckTBM_ = false;
}


// ----------------------------------------------------------------------
SiPixelRocStatus::~SiPixelRocStatus() {

}

// ----------------------------------------------------------------------
void SiPixelRocStatus::fillDIGI() {

  fDC++;

}

// ----------------------------------------------------------------------
void SiPixelRocStatus::updateDIGI(unsigned int hits) {

  fDC += hits;

}

// ----------------------------------------------------------------------
void SiPixelRocStatus::fillStuckTBM(){

     isStuckTBM_ = true;

}

// ----------------------------------------------------------------------
unsigned int SiPixelRocStatus::digiOccROC() {

  return fDC;

}

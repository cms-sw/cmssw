#include <iostream>
#include <map>
#include <cmath>
#include <vector>

#include "CalibTracker/SiPixelQuality/interface/SiPixelRocStatus.h"

using namespace std;

// ----------------------------------------------------------------------
SiPixelRocStatus::SiPixelRocStatus() {

  for (int i = 0; i < nDC_; ++i) {
       fDC[i] = 0;
  }
  isStuckTBM_ = false;
  badFed_ = -1;
  badLink_ = -1;
  startBadTime_ = -1;
  badFreq_ = 0;
}


// ----------------------------------------------------------------------
SiPixelRocStatus::~SiPixelRocStatus() {

}

// ----------------------------------------------------------------------
void SiPixelRocStatus::fillDIGI(int idc) {

  if (idc<nDC_) fDC[idc]++;

}

// ----------------------------------------------------------------------
void SiPixelRocStatus::updateDIGI(int idc, unsigned long int hits) {

  if (idc<nDC_) fDC[idc] += hits;

}

// ----------------------------------------------------------------------
void SiPixelRocStatus::fillStuckTBM(unsigned int fed, unsigned int link, std::time_t time){

     isStuckTBM_ = true;
     if(badFreq_==0){
        startBadTime_ = time;
     }
     badFed_ = fed; badLink_ = link;
     badFreq_ = badFreq_ + 1;
}

void SiPixelRocStatus::updateStuckTBM(unsigned int fed, unsigned int link, std::time_t time, unsigned long int freq){

     isStuckTBM_ = true;
     if(badFreq_==0){
        startBadTime_ = time;
     }
     badFed_ = fed; badLink_ = link;
     badFreq_ = badFreq_ + freq;
}

// ----------------------------------------------------------------------
unsigned long int SiPixelRocStatus::digiOccDC(int idc) {

  return (idc<nDC_?fDC[idc]:-1);

}

// ----------------------------------------------------------------------
unsigned long int SiPixelRocStatus::digiOccROC() {

  unsigned long int count(0) ;
  for (int idc = 0; idc < nDC_; ++idc) {
    count += fDC[idc];
  }
  return count;
}


#include <iostream>
#include <map>
#include <cmath>
#include <vector>

#include <TMath.h>
#include <TH1.h>

#include "CalibTracker/SiPixelQuality/interface/SiPixelModuleStatus.h"

using namespace std;

// ----------------------------------------------------------------------
SiPixelModuleStatus::SiPixelModuleStatus(int detId, int nrocs): fDetid(detId), fNrocs(nrocs) {

  for (int i = 0; i < fNrocs; ++i) {
    SiPixelRocStatus a;
    fRocs.push_back(a);
  }

  fModAverage = fModSigma = 0.;

};


// ----------------------------------------------------------------------
SiPixelModuleStatus::~SiPixelModuleStatus() {};


// ----------------------------------------------------------------------
void SiPixelModuleStatus::fillDIGI(int iroc) {

  if (iroc < fNrocs) 
     fRocs[iroc].fillDIGI();

}

// ----------------------------------------------------------------------
void SiPixelModuleStatus::updateDIGI(int iroc, unsigned int nhit) {

  if (iroc < fNrocs) 
     fRocs[iroc].updateDIGI(nhit);

}

// ----------------------------------------------------------------------
void SiPixelModuleStatus::fillStuckTBM( PixelFEDChannel ch){

     int roc_first = int(ch.roc_first); int roc_last = int(ch.roc_last);
     for (int iroc = 0; iroc < fNrocs; iroc++){
          if(iroc>=roc_first && iroc<=roc_last){
             fRocs[iroc].fillStuckTBM();
          }
     }

}

// ----------------------------------------------------------------------
unsigned int SiPixelModuleStatus::digiOccROC(int iroc) {

  return (iroc < fNrocs ? fRocs[iroc].digiOccROC() : -1);

}

// ----------------------------------------------------------------------
unsigned int SiPixelModuleStatus::digiOccMOD() {

  unsigned int count(0);
  for (int iroc = 0; iroc < fNrocs; ++iroc) {
    count += digiOccROC(iroc);
  }
  return count;

}

// ----------------------------------------------------------------------
int SiPixelModuleStatus::detid() {

  return fDetid;

}

// ----------------------------------------------------------------------
int SiPixelModuleStatus::nrocs() {

  return fNrocs;

}

// ----------------------------------------------------------------------
void SiPixelModuleStatus::setNrocs(int iroc) {

  fNrocs = iroc;

}


// ----------------------------------------------------------------------
double SiPixelModuleStatus::perRocDigiOcc() {

  digiOccupancy();
  return fModAverage;

}


// ----------------------------------------------------------------------
double SiPixelModuleStatus::perRocDigiOccVar() {

  digiOccupancy();
  return fModSigma;

}

// ----------------------------------------------------------------------
void SiPixelModuleStatus::digiOccupancy() {

  fModAverage = fModSigma = 0.;
  unsigned int ave(0), sig(0);
  for (int iroc = 0; iroc < fNrocs; ++iroc) {
    unsigned int inc = digiOccROC(iroc);
    ave += inc;
  }
  fModAverage = (1.0*ave)/fNrocs;

  for (int iroc = 0; iroc < fNrocs; ++iroc) {
    unsigned int inc = digiOccROC(iroc);
    sig += (fModAverage-inc)*(fModAverage-inc);
  }

  fModSigma   = sig/(fNrocs-1);
  fModSigma   = TMath::Sqrt(fModSigma);

}

// ----------------------------------------------------------------------
// Be careful : return the address not the value of ROC status
SiPixelRocStatus* SiPixelModuleStatus::getRoc(int i) {

  return &fRocs[i];

}

// ----------------------------------------------------------------------
void SiPixelModuleStatus::updateModuleDIGI(int iroc, unsigned int nhits) {

     fRocs[iroc].updateDIGI(nhits);

}

void SiPixelModuleStatus::updateModuleStatus(SiPixelModuleStatus newData) {

     bool isSameModule = true;
     if( fDetid!=newData.detid() || fNrocs!=newData.nrocs()) {
         isSameModule = false;
     }

     if(isSameModule){

        for (int iroc = 0; iroc < fNrocs; ++iroc) {
             //update occupancy
             fRocs[iroc].updateDIGI(newData.digiOccROC(iroc));
             //update stuckTBM
             SiPixelRocStatus* rocStatus = newData.getRoc(iroc);
             if(rocStatus->isStuckTBM()){
                 fRocs[iroc].fillStuckTBM();
             }
        }
        
     } // if same module

}

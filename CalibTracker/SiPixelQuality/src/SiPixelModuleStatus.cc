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
void SiPixelModuleStatus::fillDIGI(int iroc, int idc) {

  if (iroc < fNrocs) 
     fRocs[iroc].fillDIGI(idc);

}

// ----------------------------------------------------------------------
void SiPixelModuleStatus::updateDIGI(int iroc, int idc, unsigned long int nhit) {

  if (iroc < fNrocs) 
     fRocs[iroc].updateDIGI(idc, nhit);

}

// ----------------------------------------------------------------------
void SiPixelModuleStatus::fillStuckTBM( PixelFEDChannel ch, std::time_t time ){

     int fed = int(ch.fed);
     int link = int(ch.link);
     int roc_first = int(ch.roc_first); int roc_last = int(ch.roc_last);
     for (int iroc = 0; iroc < fNrocs; iroc++){
          if(iroc>=roc_first && iroc<=roc_last){
             fRocs[iroc].fillStuckTBM(fed,link,time);
          }
     }

}

// ----------------------------------------------------------------------
unsigned long int SiPixelModuleStatus::digiOccDC(int iroc, int idc) {

  return (iroc < fNrocs ? fRocs[iroc].digiOccDC(idc) : -1);

}


// ----------------------------------------------------------------------
unsigned long int SiPixelModuleStatus::digiOccROC(int iroc) {

  return (iroc < fNrocs ? fRocs[iroc].digiOccROC() : -1);

}


// ----------------------------------------------------------------------
unsigned long int SiPixelModuleStatus::digiOccMOD() {

  unsigned long int count(0);
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
  unsigned long int ave(0.), sig(0.);
  for (int iroc = 0; iroc < fNrocs; ++iroc) {
    unsigned long int inc = digiOccROC(iroc);
    ave += inc;
  }
  fModAverage = (1.0*ave)/fNrocs;

  for (int iroc = 0; iroc < fNrocs; ++iroc) {
    unsigned long int inc = digiOccROC(iroc);
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
void SiPixelModuleStatus::updateModuleDIGI(int iroc, int idc, unsigned long int nhits) {
     fRocs[iroc].updateDIGI(idc,nhits);
}

void SiPixelModuleStatus::updateModuleStatus(SiPixelModuleStatus newData) {

     bool isSameModule = true;
     if( fDetid!=newData.detid() || fNrocs!=newData.nrocs()) {
         isSameModule = false;
     }

     if(isSameModule){

        for (int iroc = 0; iroc < fNrocs; ++iroc) {
             int nDC = fRocs[iroc].nDC();
             //update occupancy
             for(int idc = 0; idc < nDC; ++idc) {
                 fRocs[iroc].updateDIGI(idc,newData.digiOccDC(iroc,idc));
             }
             //update stuckTBM
             SiPixelRocStatus* rocStatus = newData.getRoc(iroc);
             if(rocStatus->isStuckTBM()){
                 fRocs[iroc].updateStuckTBM(rocStatus->getBadFed(), rocStatus->getBadLink(),rocStatus->getStartBadTime(), rocStatus->getBadFreq());
             }
        }
        
     } // if same module

}

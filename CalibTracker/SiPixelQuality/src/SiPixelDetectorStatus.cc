#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <vector>

#include <TString.h>
#include <TMath.h>
#include <TH1.h>

#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"

// ----------------------------------------------------------------------
SiPixelDetectorStatus::SiPixelDetectorStatus(): fLS0(99999999), fLS1(0), fRun0(99999999), fRun1(0) {

  fDetHits = 0;
  fNevents = 0;

}

// ----------------------------------------------------------------------
SiPixelDetectorStatus::~SiPixelDetectorStatus() {

}


// ----------------------------------------------------------------------
void SiPixelDetectorStatus::readFromFile(std::string filename) {

  std::ifstream INS;
  std::string sline;
  INS.open(filename.c_str());

  int oldDetId(-1);
  int detid(0), roc(0), hits(0), nroc(0);
  SiPixelModuleStatus *pMod(nullptr);
  bool readOK(false);
  while (std::getline(INS, sline)) {

    if (std::string::npos != sline.find("# SiPixelDetectorStatus START")) {
      readOK = true;
      continue;
    }
    if (!readOK) continue;

    if (std::string::npos != sline.find("# SiPixelDetectorStatus END")) {
      pMod->setNrocs(nroc+1);
      break;
    }

    if (sline.find("# SiPixelDetectorStatus for LS") != std::string::npos) {
      std::sscanf(sline.c_str(), "# SiPixelDetectorStatus for LS  %d .. %d", &fLS0, &fLS1);
      continue;
    }
    if (sline.find("# SiPixelDetectorStatus for run") != std::string::npos) {
      std::sscanf(sline.c_str(), "# SiPixelDetectorStatus for run %d .. %d", &fRun0, &fRun1);
      continue;
    }
    if (sline.find("# SiPixelDetectorStatus total hits = ") != std::string::npos) {
      std::sscanf(sline.c_str(), "# SiPixelDetectorStatus total hits = %ld", &fDetHits);
      continue;
    }

    std::sscanf(sline.c_str(), "%d %d %d", &detid, &roc, &hits);
    if (roc > nroc) nroc = roc;
    if (detid != oldDetId) {
      if (pMod) {
	pMod->setNrocs(nroc+1);
      }

      oldDetId = detid;
      if (getModule(detid) == nullptr) {
	addModule(detid,nroc+1);
      } 

      pMod = getModule(detid);
      nroc = 0;
    }
    if (pMod) {
      fDetHits += hits;
      pMod->updateModuleDIGI(roc, hits);
    }

  }

  INS.close();

}


// ----------------------------------------------------------------------
void SiPixelDetectorStatus::dumpToFile(std::string filename) {

  std::ofstream OD(filename.c_str());
  OD << "# SiPixelDetectorStatus START" << std::endl;
  OD << "# SiPixelDetectorStatus for LS  " << fLS0 << " .. " << fLS1 << std::endl;
  OD << "# SiPixelDetectorStatus for run " << fRun0 << " .. " << fRun1 << std::endl;
  OD << "# SiPixelDetectorStatus total hits = " << fDetHits << std::endl;

  for (std::map<int, SiPixelModuleStatus>::iterator it = SiPixelDetectorStatus::begin(); it != SiPixelDetectorStatus::end(); ++it) {
    for (int iroc = 0; iroc < it->second.nrocs(); ++iroc) {
      for (int idc = 0; idc < 26; ++idc) {
	OD << Form("%10d %2d %3d", it->first, iroc, int(it->second.getRoc(iroc)->digiOccROC())) << std::endl;
      }
    }
  }
  OD << "# SiPixelDetectorStatus END" << std::endl;
  OD.close();

}


// ----------------------------------------------------------------------
void SiPixelDetectorStatus::addModule(int detid, int nrocs) {

     SiPixelModuleStatus a(detid, nrocs);
     fModules.insert(std::make_pair(detid, a));

}

// ----------------------------------------------------------------------
void SiPixelDetectorStatus::addModule(int detid, SiPixelModuleStatus a) {

     fModules.insert(std::make_pair(detid, a));

}


// ----------------------------------------------------------------------
void SiPixelDetectorStatus::fillDIGI(int detid, int roc) {

     ++fDetHits;
     fModules[detid].fillDIGI(roc);

}

// ----------------------------------------------------------------------
void SiPixelDetectorStatus::fillFEDerror25(int detid,PixelFEDChannel ch){

   if (fModules.find(detid) != fModules.end()){
        fModules[detid].fillFEDerror25(ch);
   }

}

// FEDerror25 effected ROCs in for each module
std::map<int, std::vector<int>> SiPixelDetectorStatus::getFEDerror25Rocs(){

    std::map<int, std::vector<int>> badRocLists_;

    for(std::map<int, SiPixelModuleStatus>::iterator itMod = SiPixelDetectorStatus::begin(); itMod != SiPixelDetectorStatus::end(); ++itMod)
    {
         int detid = itMod->first;
         // FEDerror25 effected ROCs in a given module
         std::vector<int> list;
         SiPixelModuleStatus modStatus = itMod->second;
         for (int iroc = 0; iroc < modStatus.nrocs(); ++iroc) {

              SiPixelRocStatus* roc = modStatus.getRoc(iroc);
              if(roc->isFEDerror25()){
                 list.push_back(iroc);
              }
              badRocLists_[detid]=list;
         }

    }

    return badRocLists_;
}

// ----------------------------------------------------------------------
std::map<int, SiPixelModuleStatus>::iterator SiPixelDetectorStatus::begin() {

  return fModules.begin();

}

// ----------------------------------------------------------------------
//map<int, SiPixelModuleStatus>::iterator SiPixelDetectorStatus::next() {
//  return fNext++;
//}

// ----------------------------------------------------------------------
std::map<int, SiPixelModuleStatus>::iterator SiPixelDetectorStatus::end() {

  return fModules.end();

}

// ----------------------------------------------------------------------
int SiPixelDetectorStatus::nmodules() {

  return static_cast<int>(fModules.size());

}

// ----------------------------------------------------------------------
SiPixelModuleStatus* SiPixelDetectorStatus::getModule(int detid) {

  if (fModules.find(detid) == fModules.end()) {
    return nullptr;
  }
  return &(fModules[detid]);

}

bool SiPixelDetectorStatus::findModule(int detid) {

  if (fModules.find(detid) == fModules.end()) 
    return false;
  else
    return true;

}

// ----------------------------------------------------------------------
double SiPixelDetectorStatus::perRocDigiOcc() {

  unsigned long int ave(0);
  int nrocs(0);
  for (std::map<int, SiPixelModuleStatus>::iterator it = SiPixelDetectorStatus::begin(); it != SiPixelDetectorStatus::end(); ++it) {
    unsigned long int inc = it->second.digiOccMOD();
    ave   += inc;
    nrocs += it->second.nrocs();
  }
  return (1.0*ave)/nrocs;

}

double SiPixelDetectorStatus::perRocDigiOccVar(){

  double fDetAverage = SiPixelDetectorStatus::perRocDigiOcc();

  double sig = 0.0;
  int nrocs(0);
  for(std::map<int, SiPixelModuleStatus>::iterator it = SiPixelDetectorStatus::begin(); it != SiPixelDetectorStatus::end(); ++it) {
    unsigned long int inc = it->second.digiOccMOD();
    sig += (fDetAverage - inc) * (fDetAverage - inc);
    nrocs += it->second.nrocs();
  }

  double fDetSigma   = sig/(nrocs - 1);
  return TMath::Sqrt(fDetSigma);
}

// combine status from different data (coming from different run/lumi)
void SiPixelDetectorStatus::updateDetectorStatus(SiPixelDetectorStatus newData){

  // loop over new data status
  for(std::map<int, SiPixelModuleStatus>::iterator it = newData.begin(); it != newData.end(); ++it) {
       
       int detid = it->first;
       if(fModules.find(detid) != fModules.end()){// if the detid is in the module lists
          fModules[detid].updateModuleStatus( *(newData.getModule(detid)) );
       }
       else{
          fModules.insert(std::make_pair(detid, *(newData.getModule(detid))));
       }

  }

  fDetHits = fDetHits + newData.digiOccDET();
  fNevents = fNevents + newData.getNevents();

}

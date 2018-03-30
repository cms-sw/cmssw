#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <vector>

#include <TString.h>
#include <TMath.h>
#include <TH1.h>

#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;


// ----------------------------------------------------------------------
SiPixelDetectorStatus::SiPixelDetectorStatus(): fLS0(99999999), fLS1(0), fRun0(99999999), fRun1(0), fDetHits(0) {

  fDetAverage = fDetSigma = 0.;

  fTime0 = fTime1 = 0;
  fNevents = 0;

}

// ----------------------------------------------------------------------
SiPixelDetectorStatus::~SiPixelDetectorStatus() {

}


// ----------------------------------------------------------------------
void SiPixelDetectorStatus::readFromFile(std::string filename) {

  ifstream INS;
  string sline;
  INS.open(filename.c_str());

  int oldDetId(-1);
  int detid(0), roc(0), dc(0), hits(0), nroc(0);
  SiPixelModuleStatus *pMod(nullptr);
  bool readOK(false);
  while (getline(INS, sline)) {

    if (string::npos != sline.find("# SiPixelDetectorStatus START")) {
      readOK = true;
      continue;
    }
    if (!readOK) continue;

    if (string::npos != sline.find("# SiPixelDetectorStatus END")) {
      pMod->setNrocs(nroc+1);
      break;
    }

    if (sline.find("# SiPixelDetectorStatus for LS") != string::npos) {
      sscanf(sline.c_str(), "# SiPixelDetectorStatus for LS  %d .. %d", &fLS0, &fLS1);
      continue;
    }
    if (sline.find("# SiPixelDetectorStatus for run") != string::npos) {
      sscanf(sline.c_str(), "# SiPixelDetectorStatus for run %d .. %d", &fRun0, &fRun1);
      continue;
    }
    if (sline.find("# SiPixelDetectorStatus total hits = ") != string::npos) {
      sscanf(sline.c_str(), "# SiPixelDetectorStatus total hits = %ld", &fDetHits);
      continue;
    }

    sscanf(sline.c_str(), "%d %d %d %d", &detid, &roc, &dc, &hits);
    if (roc > nroc) nroc = roc;
    if (detid != oldDetId) {
      if (pMod) {
	pMod->setNrocs(nroc+1);
      }

      oldDetId = detid;
      if (nullptr == getModule(detid)) {
	addModule(detid,nroc+1);
      } 

      pMod = getModule(detid);
      nroc = 0;
    }
    if (pMod) {
      fDetHits += hits;
      pMod->updateModuleDIGI(roc, dc, hits);
    }

  }

  INS.close();

}


// ----------------------------------------------------------------------
void SiPixelDetectorStatus::dumpToFile(std::string filename) {

  ofstream OD(filename.c_str());
  OD << "# SiPixelDetectorStatus START" << endl;
  OD << "# SiPixelDetectorStatus for LS  " << fLS0 << " .. " << fLS1 << endl;
  OD << "# SiPixelDetectorStatus for run " << fRun0 << " .. " << fRun1 << endl;
  OD << "# SiPixelDetectorStatus total hits = " << fDetHits << endl;
  map<int, SiPixelModuleStatus>::iterator itEnd = end();
  for (map<int, SiPixelModuleStatus>::iterator it = begin(); it != itEnd; ++it) {
    for (int iroc = 0; iroc < it->second.nrocs(); ++iroc) {
      for (int idc = 0; idc < 26; ++idc) {
	OD << Form("%10d %2d %2d %3d", it->first, iroc, idc, int(it->second.getRoc(iroc)->digiOccDC(idc))) << endl;
      }
    }
  }
  OD << "# SiPixelDetectorStatus END" << endl;
  OD.close();

}


// ----------------------------------------------------------------------
void SiPixelDetectorStatus::addModule(int detid, int nrocs) {
     SiPixelModuleStatus a(detid, nrocs);
     fModules.insert(make_pair(detid, a));
}
// ----------------------------------------------------------------------
void SiPixelDetectorStatus::addModule(int detid, SiPixelModuleStatus a) {
     fModules.insert(make_pair(detid, a));
}


// ----------------------------------------------------------------------
void SiPixelDetectorStatus::fillDIGI(int detid, int roc, int idc) {
     ++fDetHits;
     fModules[detid].fillDIGI(roc, idc);
}

// ----------------------------------------------------------------------
void SiPixelDetectorStatus::fillStuckTBM(int detid,PixelFEDChannel ch, std::time_t time){

   if (fModules.find(detid) != fModules.end()){
        fModules[detid].fillStuckTBM(ch,time);
   }

}

// stuck TBM effected ROCs in for each module
std::map<int, std::vector<int>> SiPixelDetectorStatus::getStuckTBMsRocs(){

    std::map<int, std::vector<int>> badRocLists_;

    std::map<int, SiPixelModuleStatus>::iterator itModEnd = end();
    for (std::map<int, SiPixelModuleStatus>::iterator itMod = begin(); itMod != itModEnd; ++itMod) {

         int detid = itMod->first;
         // stuck TBM effected ROCs in a given module
         std::vector<int> list;
         SiPixelModuleStatus modStatus = itMod->second;
         for (int iroc = 0; iroc < modStatus.nrocs(); ++iroc) {

              SiPixelRocStatus* roc = modStatus.getRoc(iroc);
              if(roc->isStuckTBM()){
                 list.push_back(iroc);
              }
              badRocLists_[detid]=list;
         }

    }

    return badRocLists_;
}

// ----------------------------------------------------------------------
map<int, SiPixelModuleStatus>::iterator SiPixelDetectorStatus::begin() {
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
void SiPixelDetectorStatus::digiOccupancy() {

  fDetAverage = fDetSigma = 0;
  unsigned long int ave(0), sig(0);
  int nrocs(0);
  map<int, SiPixelModuleStatus>::iterator itEnd = end();
  for (map<int, SiPixelModuleStatus>::iterator it = begin(); it != itEnd; ++it) {
    unsigned long int inc = it->second.digiOccMOD();
    ave   += inc;
    nrocs += it->second.nrocs();
  }
  fDetAverage = (1.0*ave)/nrocs;

  for (map<int, SiPixelModuleStatus>::iterator it = begin(); it != itEnd; ++it) {
    unsigned long int inc = it->second.digiOccMOD();
    sig += (fDetAverage - inc) * (fDetAverage - inc);
  }

  fDetSigma   = sig/(nrocs - 1);
  fDetSigma   = TMath::Sqrt(fDetSigma);

}

// combine status from different data (coming from different run/lumi)
void SiPixelDetectorStatus::updateDetectorStatus(SiPixelDetectorStatus newData){

  // loop over new data status
  std::map<int, SiPixelModuleStatus>::iterator itEnd = newData.end();
  for (map<int, SiPixelModuleStatus>::iterator it = newData.begin(); it != itEnd; ++it) {
       
       int detid = it->first;
       if(fModules.find(detid) != fModules.end()){// if the detid is in the module lists
          fModules[detid].updateModuleStatus( *(newData.getModule(detid)) );
       }

  }

  fDetHits = fDetHits + newData.digiOccDET();
  fNevents = fNevents + newData.getNevents();

}

SiPixelDetectorStatus SiPixelDetectorStatus::combineDetectorStatus(SiPixelDetectorStatus newData){

  SiPixelDetectorStatus combine;

  // loop over current module status
  std::map<int, SiPixelModuleStatus>::iterator itEnd = end();
  for (map<int, SiPixelModuleStatus>::iterator it = begin(); it != itEnd; ++it) {

       int detid = it->first;
       combine.addModule(detid, fModules[detid]);

       if(newData.findModule(detid)){ // the detid in current data is also in new data
          // then update the module status
          SiPixelModuleStatus* moduleStatus = combine.getModule(detid);
          moduleStatus->updateModuleStatus(*(newData.getModule(detid)));
       }
  } 

  return combine; 

}

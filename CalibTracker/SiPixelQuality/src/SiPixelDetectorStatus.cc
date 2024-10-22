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
SiPixelDetectorStatus::SiPixelDetectorStatus() : fLS0_(99999999), fLS1_(0), fRun0_(99999999), fRun1_(0) {
  fDetHits_ = 0;
  ftotalevents_ = 0;
}

// ----------------------------------------------------------------------
SiPixelDetectorStatus::~SiPixelDetectorStatus() {}

// ----------------------------------------------------------------------
void SiPixelDetectorStatus::readFromFile(std::string filename) {
  std::ifstream INS;
  std::string sline;
  INS.open(filename.c_str());

  int oldDetId(-1);
  int detid(0), roc(0), hits(0), nroc(0);
  SiPixelModuleStatus* pMod(nullptr);
  bool readOK(false);
  while (std::getline(INS, sline)) {
    if (std::string::npos != sline.find("# SiPixelDetectorStatus START")) {
      readOK = true;
      continue;
    }
    if (!readOK)
      continue;

    if (std::string::npos != sline.find("# SiPixelDetectorStatus END")) {
      pMod->setNrocs(nroc + 1);
      break;
    }

    if (sline.find("# SiPixelDetectorStatus for LS") != std::string::npos) {
      std::sscanf(sline.c_str(), "# SiPixelDetectorStatus for LS  %d .. %d", &fLS0_, &fLS1_);
      continue;
    }
    if (sline.find("# SiPixelDetectorStatus for run") != std::string::npos) {
      std::sscanf(sline.c_str(), "# SiPixelDetectorStatus for run %d .. %d", &fRun0_, &fRun1_);
      continue;
    }
    if (sline.find("# SiPixelDetectorStatus total hits = ") != std::string::npos) {
      std::sscanf(sline.c_str(), "# SiPixelDetectorStatus total hits = %ld", &fDetHits_);
      continue;
    }

    std::sscanf(sline.c_str(), "%d %d %d", &detid, &roc, &hits);
    if (roc > nroc)
      nroc = roc;
    if (detid != oldDetId) {
      if (pMod) {
        pMod->setNrocs(nroc + 1);  // roc ranges from 0, "+1" to get number of rocs per module
      }

      oldDetId = detid;
      if (getModule(detid) == nullptr) {
        addModule(detid, nroc + 1);  // roc ranges from 0, "+1" to get number of rocs per module
      }

      pMod = getModule(detid);
      nroc = 0;
    }
    // for existing module, update its content
    if (pMod != nullptr) {
      fDetHits_ += hits;
      pMod->updateModuleDIGI(roc, hits);
    }
  }

  INS.close();
}

// ----------------------------------------------------------------------
void SiPixelDetectorStatus::dumpToFile(std::ofstream& OD) {
  OD << "# SiPixelDetectorStatus START" << std::endl;
  OD << "# SiPixelDetectorStatus for LS  " << fLS0_ << " .. " << fLS1_ << std::endl;
  OD << "# SiPixelDetectorStatus for run " << fRun0_ << " .. " << fRun1_ << std::endl;
  OD << "# SiPixelDetectorStatus total hits = " << fDetHits_ << std::endl;

  for (std::map<int, SiPixelModuleStatus>::iterator it = SiPixelDetectorStatus::begin();
       it != SiPixelDetectorStatus::end();
       ++it) {
    for (int iroc = 0; iroc < it->second.nrocs(); ++iroc) {
      OD << Form("%10d %2d %3d", it->first, iroc, int(it->second.getRoc(iroc)->digiOccROC())) << std::endl;
    }
  }
  OD << "# SiPixelDetectorStatus END" << std::endl;
}

// ----------------------------------------------------------------------
void SiPixelDetectorStatus::addModule(int detid, int nrocs) {
  // only need to add NEW modules
  if (fModules_.find(detid) == fModules_.end()) {
    SiPixelModuleStatus a(detid, nrocs);
    fModules_.insert(std::make_pair(detid, a));
  }
}

// ----------------------------------------------------------------------
void SiPixelDetectorStatus::addModule(int detid, SiPixelModuleStatus a) { fModules_.insert(std::make_pair(detid, a)); }

// ----------------------------------------------------------------------
void SiPixelDetectorStatus::fillDIGI(int detid, int roc) {
  ++fDetHits_;
  fModules_[detid].fillDIGI(roc);
}

// ----------------------------------------------------------------------
void SiPixelDetectorStatus::fillFEDerror25(int detid, PixelFEDChannel ch) {
  if (fModules_.find(detid) != fModules_.end()) {
    fModules_[detid].fillFEDerror25(ch);
  }
}

// FEDerror25 effected ROCs in for each module
std::map<int, std::vector<int>> SiPixelDetectorStatus::getFEDerror25Rocs() {
  std::map<int, std::vector<int>> badRocLists;

  for (std::map<int, SiPixelModuleStatus>::iterator itMod = SiPixelDetectorStatus::begin();
       itMod != SiPixelDetectorStatus::end();
       ++itMod) {
    int detid = itMod->first;
    // FEDerror25 effected ROCs in a given module
    std::vector<int> list;
    SiPixelModuleStatus modStatus = itMod->second;
    for (int iroc = 0; iroc < modStatus.nrocs(); ++iroc) {
      SiPixelRocStatus* roc = modStatus.getRoc(iroc);
      if (roc->isFEDerror25()) {
        list.push_back(iroc);
      }
      badRocLists[detid] = list;
    }
  }

  return badRocLists;
}

// ----------------------------------------------------------------------
std::map<int, SiPixelModuleStatus>::iterator SiPixelDetectorStatus::begin() { return fModules_.begin(); }

// ----------------------------------------------------------------------
std::map<int, SiPixelModuleStatus>::iterator SiPixelDetectorStatus::end() { return fModules_.end(); }

// ----------------------------------------------------------------------
int SiPixelDetectorStatus::nmodules() { return static_cast<int>(fModules_.size()); }

// ----------------------------------------------------------------------
SiPixelModuleStatus* SiPixelDetectorStatus::getModule(int detid) {
  if (fModules_.find(detid) == fModules_.end()) {
    return nullptr;
  }
  return &(fModules_[detid]);
}

bool SiPixelDetectorStatus::findModule(int detid) {
  if (fModules_.find(detid) == fModules_.end())
    return false;
  else
    return true;
}

// ----------------------------------------------------------------------
double SiPixelDetectorStatus::perRocDigiOcc() {
  unsigned long int ave(0);
  int nrocs(0);
  for (std::map<int, SiPixelModuleStatus>::iterator it = SiPixelDetectorStatus::begin();
       it != SiPixelDetectorStatus::end();
       ++it) {
    unsigned long int inc = it->second.digiOccMOD();
    ave += inc;
    nrocs += it->second.nrocs();
  }
  return (1.0 * ave) / nrocs;
}

double SiPixelDetectorStatus::perRocDigiOccVar() {
  double fDetAverage = SiPixelDetectorStatus::perRocDigiOcc();

  double sig = 0.0;
  int nrocs(0);
  for (std::map<int, SiPixelModuleStatus>::iterator it = SiPixelDetectorStatus::begin();
       it != SiPixelDetectorStatus::end();
       ++it) {
    unsigned long int inc = it->second.digiOccMOD();
    sig += (fDetAverage - inc) * (fDetAverage - inc);
    nrocs += it->second.nrocs();
  }

  double fDetSigma = sig / (nrocs - 1);
  return TMath::Sqrt(fDetSigma);
}

/*|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||*/

// combine status from different data (coming from different run/lumi)
void SiPixelDetectorStatus::updateDetectorStatus(SiPixelDetectorStatus newData) {
  // loop over new data status
  for (std::map<int, SiPixelModuleStatus>::iterator it = newData.begin(); it != newData.end(); ++it) {
    int detid = it->first;

    if (fModules_.find(detid) != fModules_.end()) {  // if the detid is in the module lists
      fModules_[detid].updateModuleStatus(*(newData.getModule(detid)));
    } else {  // if new module, add(insert) the module data
      fModules_.insert(std::make_pair(detid, *(newData.getModule(detid))));
    }
  }

  fDetHits_ = fDetHits_ + newData.digiOccDET();
  ftotalevents_ = ftotalevents_ + newData.getNevents();
}

void SiPixelDetectorStatus::resetDetectorStatus() {
  fModules_.clear();
  fDetHits_ = 0;
  ftotalevents_ = 0;
  fRun0_ = 99999999;
  fRun1_ = 0;
  fLS0_ = 99999999;
  fLS1_ = 0;
}

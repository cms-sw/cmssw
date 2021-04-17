#ifndef SIPIXELDETECTORSTATUS_h
#define SIPIXELDETECTORSTATUS_h

#include <ctime>
#include <map>
#include <string>

#include "CalibTracker/SiPixelQuality/interface/SiPixelModuleStatus.h"

// ----------------------------------------------------------------------
class SiPixelDetectorStatus {
public:
  SiPixelDetectorStatus();
  ~SiPixelDetectorStatus();

  // reset
  void resetDetectorStatus();
  // combine detector status
  void updateDetectorStatus(SiPixelDetectorStatus newData);

  // file I/O
  void readFromFile(std::string filename);
  void dumpToFile(std::ofstream& outFile);

  /*|||||||||||||||||||||||||||||||||||||||||||||||||||||||*/

  // add SiPixelModuleStatus for detID, specifying nrocs
  void addModule(int detid, int nrocs);
  // add a SiPixelModuleStatus obj for detID
  void addModule(int detid, SiPixelModuleStatus a);
  // get a Module
  bool findModule(int detid);
  SiPixelModuleStatus* getModule(int detid);

  // fill hit in double idc in ROC roc into module detid
  void fillDIGI(int detid, int roc);
  // fill FEDerror25 info
  void fillFEDerror25(int detid, PixelFEDChannel ch);

  // detector status : std:map - collection of module status
  std::map<int, SiPixelModuleStatus> getDetectorStatus() { return fModules_; }
  // list of ROCs with FEDerror25
  std::map<int, std::vector<int>> getFEDerror25Rocs();
  // total number of DIGIs
  unsigned long int digiOccDET() { return fDetHits_; }
  // total processed events
  void setNevents(unsigned long int N) { ftotalevents_ = N; }
  unsigned long int getNevents() { return ftotalevents_; }

  // number of modules in detector
  int nmodules();
  // determine detector average nhits and RMS
  double perRocDigiOcc();
  double perRocDigiOccVar();

  // set the time stamps
  void setRunRange(int run0, int run1) {
    fRun0_ = run0;
    fRun1_ = run1;
  }
  std::pair<int, int> getRunRange() { return std::make_pair(fRun0_, fRun1_); }
  //////////////////////////////////////////////////////////////////////////////////
  void setLSRange(int ls0, int ls1) {
    fLS0_ = ls0;
    fLS1_ = ls1;
  }
  std::pair<int, int> getLSRange() { return std::make_pair(fLS0_, fLS1_); }

  // provide for iterating over the entire detector
  std::map<int, SiPixelModuleStatus>::iterator begin();
  std::map<int, SiPixelModuleStatus>::iterator next();
  std::map<int, SiPixelModuleStatus>::iterator end();

private:
  std::map<int, SiPixelModuleStatus> fModules_;

  // first and last lumisection seen in this instance
  int fLS0_, fLS1_;
  // first and last run (should be the same number! as currently only perform Single Run Harvestor)
  int fRun0_, fRun1_;

  // number of events processed
  unsigned long int ftotalevents_;

  // total hits in detector
  unsigned long int fDetHits_;
};

#endif

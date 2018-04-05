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

  // file I/O
  void readFromFile(std::string filename);
  void dumpToFile(std::string filename);

  // add SiPixelModuleStatus for detID, specifying nrocs
  void addModule(int detid, int nrocs);
  // add a SiPixelModuleStatus obj for detID
  void addModule(int detid, SiPixelModuleStatus a);

  // fill hit in double idc in ROC roc into module detid
  void fillDIGI(int detid, int roc, int idc);
  // fill stuck TBM info
  void fillStuckTBM(int detid, PixelFEDChannel ch, std::time_t time);

  std::map<int, std::vector<int>> getStuckTBMsRocs();

  // determine detector average nhits and RMS
  void digiOccupancy();
  double perRocDigiOcc(){ digiOccupancy(); return fDetAverage; }
  double perRocDigiOccVar(){ digiOccupancy(); return fDetSigma; }
  unsigned long int digiOccDET(){ return fDetHits; }

  // number of modules in detector
  int nmodules();

  // get a Module
  bool findModule(int detid);
  SiPixelModuleStatus* getModule(int detid);

  // provide for iterating over the entire detector
  std::map<int, SiPixelModuleStatus>::iterator begin();
  std::map<int, SiPixelModuleStatus>::iterator next();
  std::map<int, SiPixelModuleStatus>::iterator end();

  // set the time stamps
  void setRunRange(int run0, int run1) {fRun0 = run0;fRun1 = run1;}
  std::pair<int,int> getRunRange() {return std::make_pair(fRun0,fRun1);}
  void setLSRange(int ls0, int ls1)  {fLS0 = ls0; fLS1 = ls1;}
  std::pair<int,int> getLSRange() {return std::make_pair(fLS0,fLS1);}
  void setRefTime(std::time_t refTime0, std::time_t refTime1) {fTime0 = refTime0; fTime1 = refTime1;}
  std::pair<std::time_t,std::time_t> getRefTime() {return std::make_pair(fTime0,fTime1);}

  // total processed events
  void setNevents(unsigned long int N){ fNevents = N; }
  unsigned long int getNevents(){ return fNevents; }

  void resetDetectorStatus() { fModules.clear(); fDetAverage=0; fDetSigma=0; fDetHits=0; fNevents=0;
                               fRun0 = 99999999; fRun1 = 0; fLS0 = 99999999; fLS1 = 0; 
                               fTime0 = 0; fTime1 = 0;
                             }

  // combine detector status
  void updateDetectorStatus(SiPixelDetectorStatus newData);
  SiPixelDetectorStatus combineDetectorStatus(SiPixelDetectorStatus newData);

  // detector status
  std::map<int, SiPixelModuleStatus> getDetectorStatus(){ return fModules; }

 private:

  std::map<int, SiPixelModuleStatus> fModules;

  // first and last lumisection seen in this instance
  int fLS0, fLS1;

  // first and last run seen in this instance (likely to be the same number!)
  int fRun0, fRun1;

  // being and end time stamp
  std::time_t fTime0, fTime1;
  
  // number of events processed
  unsigned long int fNevents;

  // average (per module) number of hits over entire detector
  double fDetAverage, fDetSigma;

  // total hits in detector
  unsigned long int fDetHits;

};

#endif

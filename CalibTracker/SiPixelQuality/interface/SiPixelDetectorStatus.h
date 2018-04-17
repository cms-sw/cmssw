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
  void fillDIGI(int detid, int roc);
  // fill FEDerror25 info
  void fillFEDerror25(int detid, PixelFEDChannel ch);

  std::map<int, std::vector<int>> getFEDerror25Rocs();

  // determine detector average nhits and RMS
  double perRocDigiOcc();
  double perRocDigiOccVar();

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

  // total processed events
  void setNevents(unsigned long int N){ fNevents = N; }
  unsigned long int getNevents(){ return fNevents; }

  void resetDetectorStatus() { fModules.clear(); fDetHits=0; fNevents=0;
                               fRun0 = 99999999; fRun1 = 0; fLS0 = 99999999; fLS1 = 0; 
                             }

  // combine detector status
  void updateDetectorStatus(SiPixelDetectorStatus newData);

  // detector status
  std::map<int, SiPixelModuleStatus> getDetectorStatus(){ return fModules; }

 private:

  std::map<int, SiPixelModuleStatus> fModules;

  // first and last lumisection seen in this instance
  int fLS0, fLS1;
  // first and last run seen in this instance (should be the same number!)
  int fRun0, fRun1;

  // number of events processed
  unsigned long int fNevents;

  // total hits in detector
  unsigned long int fDetHits;

};

#endif

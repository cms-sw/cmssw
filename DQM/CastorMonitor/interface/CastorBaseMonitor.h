#ifndef DQM_CASTORMONITOR_CASTORBASEMONITOR_H
#define DQM_CASTORMONITOR_CASTORBASEMONITOR_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "TH1F.h"
#include "TH2F.h"
#include <map>
#include <iostream>
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h" //-- HcalCastorDetId
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h" //-- CastorDigiCollection
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h" //-- CastorRecHitCollection

#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrationWidths.h"
#include "CalibFormats/CastorObjects/interface/CastorCoder.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"

#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"

#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"


class CastorBaseMonitor {
public:
  CastorBaseMonitor(); 
  virtual ~CastorBaseMonitor(); 

  virtual void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  virtual void done();
  virtual void clearME();

  void setVerbosity(int verb) { fVerbosity = verb; }
  int getVerbosity() const { return fVerbosity; }
  
  void setDiagnostics(bool myval) { makeDiagnostics=myval;}
  bool getDiagnostics() const { return makeDiagnostics;}

  bool vetoCell(HcalCastorDetId id);

protected:
  
  int fVerbosity;
  bool showTiming; //-- controls whether to show timing diagnostic info
  edm::CPUTimer cpu_timer; 

  bool makeDiagnostics; //-- controls whether to make diagnostic plots

  DQMStore* m_dbe;
  //vector<std::string> hotCells_;
  std::string rootFolder_;
  std::string baseFolder_;

};

#endif

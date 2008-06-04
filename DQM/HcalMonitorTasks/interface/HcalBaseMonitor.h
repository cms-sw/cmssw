#ifndef DQM_HCALMONITORTASKS_HCALBASEMONITOR_H
#define DQM_HCALMONITORTASKS_HCALBASEMONITOR_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "TH1F.h"
#include "TH2F.h"
#include <map>

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include <iostream>

// Temporary fix:  Add this into base class until I figure why multiple inclusions are a problem -- Jeff, 23 May 2008
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"

using namespace std;
/** \class HcalBaseMonitor
  *  
  * $Date: 2008/05/27 03:08:57 $
  * $Revision: 1.10 $
  * \author W. Fisher - FNAL
  */
class HcalBaseMonitor {
public:
  HcalBaseMonitor(); 
  virtual ~HcalBaseMonitor(); 

  virtual void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  virtual void done();
  virtual void clearME();

  void setVerbosity(int verb) { fVerbosity = verb; }
  int getVerbosity() const { return fVerbosity; }
  
  void setDiagnostics(bool myval) { makeDiagnostics=myval;}
  bool getDiagnostics() const { return makeDiagnostics;}

  bool vetoCell(HcalDetId id);

protected:
  
  int fVerbosity;
  bool makeDiagnostics; // controls whether to make diagnostic plots

  DQMStore* m_dbe;
  vector<string> hotCells_;
  string rootFolder_;
  string baseFolder_;

};

#endif

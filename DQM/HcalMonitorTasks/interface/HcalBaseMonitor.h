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

using namespace std;
/** \class HcalBaseMonitor
  *  
  * $Date: 2007/12/01 09:55:42 $
  * $Revision: 1.8 $
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
  bool vetoCell(HcalDetId id);

protected:
  
  int fVerbosity;
  DQMStore* m_dbe;
  vector<string> hotCells_;
  string rootFolder_;
  string baseFolder_;

};

#endif

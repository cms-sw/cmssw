#ifndef DQM_CASTORMONITOR_CASTORPSMONITOR_H
#define DQM_CASTORMONITOR_CASTORPSMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQM/CastorMonitor/interface/CastorBunch.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//
//#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
//#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h" 
//
#include "FWCore/Framework/interface/Event.h"

class CastorPSMonitor: public CastorBaseMonitor {
public:
  CastorPSMonitor(); 
  ~CastorPSMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const CastorDigiCollection& castorDigis, const CastorDbService& conditions, vector<HcalGenericDetId> listEMap, int iBunch);
  void reset();

private:  

  bool doPerChannel_;
  // float occThresh_;
  int ievt_;

  ////---- define Monitoring elements
  struct{
    MonitorElement* meDigi_pulseBX    ; //-- pulse in bx's
  } castorDigiHists ;

  MonitorElement* meEvt_;
  map<int,MonitorElement*> PSsector;

  bool firstTime_;

  vector<NewBunch> Bunches_; //-- container for data, 1 per channel  

};

#endif

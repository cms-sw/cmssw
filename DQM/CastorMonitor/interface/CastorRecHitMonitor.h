#ifndef DQM_CASTORMONITOR_CASTORRECHITMONITOR_H
#define DQM_CASTORMONITOR_CASTORRECHITMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

class CastorRecHitMonitor: public CastorBaseMonitor {
public:
  CastorRecHitMonitor(const edm::ParameterSet& ps); 
  ~CastorRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps);
 void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);
  void processEvent(const CastorRecHitCollection& castorHits);

private:  
 int ievt_;
 float energyInEachChannel[14][16];
 std::string subsystemname;

 TH2F *h2RecHitMap;
 //MonitorElement* h2RHchan;
 //MonitorElement* h2RHvsSec;
 MonitorElement* h2RHmap;
 MonitorElement* h2RHoccmap;
 MonitorElement* h2RHentriesMap;
 MonitorElement* hRHtime;
 MonitorElement* hallchan;
};

#endif

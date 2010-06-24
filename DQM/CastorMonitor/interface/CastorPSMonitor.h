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
  void processEvent(const CastorDigiCollection& castorDigis, const CastorDbService& conditions, vector<HcalGenericDetId> listEMap, int iBunch, float  PedSigmaInChannel[14][16]);
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

  MonitorElement* DigiOccupancyMap;
  MonitorElement* reportSummary;
  MonitorElement* reportSummaryMap;
  MonitorElement* overallStatus;
  MonitorElement* ChannelSummaryMap;


  bool firstTime_;
  bool offline_;
  double numberSigma_; 
  int  numOK;
  double status;
  double statusRS;
  double fraction;
  TH2F* h_reportSummaryMap;

  double firstRegionThreshold_;
  double  secondRegionThreshold_;
  double thirdRegionThreshold_;
  double sumDigiForEachChannel [14][16];

  vector<NewBunch> Bunches_; //-- container for data, 1 per channel  


};

#endif

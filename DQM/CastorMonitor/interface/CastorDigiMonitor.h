#ifndef DQM_CASTORMONITOR_CASTORDIGIMONITOR_H
#define DQM_CASTORMONITOR_CASTORDIGIMONITOR_H

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CondFormats/CastorObjects/interface/CastorPedestal.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidth.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class CastorDigiMonitor {

public:
  CastorDigiMonitor(const edm::ParameterSet& ps); 
  ~CastorDigiMonitor(); 

 void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);
 void processEvent(const CastorDigiCollection& cast,const CastorDbService& cond);
 int ModSecToIndex(int module, int sector);
private:
  std::string subsystemname_;
  int fVerbosity;
  int ievt_;
  float Qrms_DEAD;

  MonitorElement* h2QrmsTSvsCh;
  MonitorElement* hQIErms[10];
  MonitorElement* hTSratio;
  MonitorElement* h2TSratio;
  MonitorElement* h2status;
  MonitorElement* h2digierr;
  MonitorElement* h2repsum;
  MonitorElement* h2qualityMap;
  MonitorElement* hReport;
  MonitorElement* h2QtsvsCh;
  MonitorElement *h2QmeantsvsCh;
  MonitorElement *h2QmeanMap;
  MonitorElement *hModule;
  MonitorElement *hSector;
  MonitorElement* hdigisize;
//  MonitorElement* hBunchOcc;

  int TS_MAX = 10;
  float RatioThresh1 = 0.;
  float QIEerrThreshold = 0.0001;
  double QrmsTS[224][10], QmeanTS[224][10];
  const int TSped = 0;
};

#endif

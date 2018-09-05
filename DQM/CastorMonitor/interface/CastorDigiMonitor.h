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

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

//#include "FWCore/Framework/interface/Run.h"


class CastorDigiMonitor {

public:
  CastorDigiMonitor(const edm::ParameterSet& ps); 
  ~CastorDigiMonitor(); 

 void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);
 void processEvent(edm::Event const& event, const CastorDigiCollection& cast, 
       const edm::TriggerResults& trig, const CastorDbService& cond);
 void endRun();
 void getDbData(const edm::EventSetup& iSetup);
 int ModSecToIndex(int module, int sector);
 void fillTrigRes(edm::Event const& event,const edm::TriggerResults& TrigResults,
	 double Etot);

private:
  std::string subsystemname_;
  int fVerbosity;
  int ievt_;
  float Qrms_DEAD;

  MonitorElement *hBX, *hpBXtrig;
  MonitorElement* hpTrigRes;
  MonitorElement* h2QrmsTSvsCh;
  MonitorElement* hQIErms[10];
  MonitorElement* hTSratio;
  MonitorElement* h2TSratio;
  MonitorElement* h2status;
  MonitorElement* h2digierr;
  MonitorElement* h2repsum;
  MonitorElement* h2qualityMap;
  MonitorElement* hReport;
  MonitorElement *h2QmeantsvsCh;
  MonitorElement *h2QmeanMap;
  MonitorElement *hModule;
  MonitorElement *hSector;
  MonitorElement* hdigisize;
  MonitorElement *h2towEMvsHAD;
  MonitorElement *htowE;

  int TS_MAX = 10;
  float RatioThresh1 = 0.;
  float QIEerrThreshold = 0.0001;
  double QrmsTS[224][10], QmeanTS[224][10];
  const int TSped = 0;
};

#endif

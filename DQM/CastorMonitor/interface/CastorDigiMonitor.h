#ifndef DQM_CASTORMONITOR_CASTORDIGIMONITOR_H
#define DQM_CASTORMONITOR_CASTORDIGIMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"

#include "CondFormats/CastorObjects/interface/CastorPedestal.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidth.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class CastorDigiMonitor: public CastorBaseMonitor {

public:
  CastorDigiMonitor(const edm::ParameterSet& ps); 
  ~CastorDigiMonitor(); 

//  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void setup(const edm::ParameterSet& ps);
 void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);
//  void beginRun(const edm::EventSetup& iSetup);
//  void beginRun(const edm::EventSetup& iSetup, DQMStore* dbe);
  void processEvent(const CastorDigiCollection& cast,const CastorDbService& cond, int bunch);
//  void processEvent(const CastorDigiCollection& cast,const CastorDbService& cond);

  void done();
  void reset();

protected:
  DQMStore* dbe;

private:
  std::string subsystemname_;
  int fVerbosity;
  int ievt_;

  MonitorElement* h2digierr;
  MonitorElement* h2QtsvsCh;
  MonitorElement *h2QmeantsvsCh;
  MonitorElement *h2QmeanMap;
  MonitorElement *hModule;
  MonitorElement *hSector;
  MonitorElement* hdigisize;
  MonitorElement* hBunchOcc;

/*
  void perChanHists(std::vector<HcalCastorDetId> detID, std::vector<int> capID, std::vector<float> peds,
		    std::map<HcalCastorDetId, std::map<int, MonitorElement*> > &toolP, 
		    ////// std::map<HcalCastorDetId, std::map<int, MonitorElement*> > &toolS,
		    std::string baseFolder);
//  bool doPerChannel_;
//  bool doFCpeds_;
  std::map<HcalCastorDetId, std::map<int,MonitorElement*> >::iterator meo_;
  std::vector<HcalCastorDetId> detID_;
  std::vector<int> capID_;
  std::vector<float> pedVals_;
//  std::string outputFile_;
   
  const CastorQIEShape* shape_;
  const CastorQIECoder* channelCoder_;
  //CastorCalibrations calibs_; 
//  MonitorElement* meEVT_;
  std::map<HcalCastorDetId,bool> REG;
  MonitorElement* PEDESTAL_REFS;
  MonitorElement* WIDTH_REFS;
*/
/*
  struct{
    std::map<HcalCastorDetId,std::map<int, MonitorElement*> > PEDVALS;
    std::map<HcalCastorDetId,std::map<int, MonitorElement*> > SUBVALS;
    MonitorElement* ALLPEDS;
    MonitorElement* PEDRMS;
    ////---- data from the Condition Database
    MonitorElement* PEDESTAL_REFS;
    MonitorElement* WIDTH_REFS;
  } castHists ;
*/
};

#endif

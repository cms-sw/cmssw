#ifndef DQM_CASTORMONITOR_CASTORDIGIMONITOR_H
#define DQM_CASTORMONITOR_CASTORDIGIMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"

#include "CondFormats/CastorObjects/interface/CastorPedestal.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidth.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class CastorDigiMonitor: public CastorBaseMonitor {
public:
  CastorDigiMonitor(); 
  ~CastorDigiMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void beginRun(const edm::EventSetup& iSetup);
  void processEvent(const CastorDigiCollection& cast,const CastorDbService& cond);


  void done();
  void reset();

private: 
  void perChanHists(const std::vector<HcalCastorDetId>& detID, const std::vector<int>& capID, const std::vector<float>& peds,
		    std::map<HcalCastorDetId, std::map<int, MonitorElement*> > &toolP, 
		    ////// std::map<HcalCastorDetId, std::map<int, MonitorElement*> > &toolS,
		    std::string baseFolder);

  bool doPerChannel_;
  bool doFCpeds_;
  std::map<HcalCastorDetId, std::map<int,MonitorElement*> >::iterator meo_;
  std::vector<HcalCastorDetId> detID_;
  std::vector<int> capID_;
  std::vector<float> pedVals_;

  std::string outputFile_;
   
  const CastorQIEShape* shape_;
  const CastorQIECoder* channelCoder_;
  //CastorCalibrations calibs_; 
  

  MonitorElement* meEVT_;
  int ievt_;
    
  std::map<HcalCastorDetId,bool> REG;
  MonitorElement* PEDESTAL_REFS;
  MonitorElement* WIDTH_REFS;

  MonitorElement* h2digierr;

  struct{
    std::map<HcalCastorDetId,std::map<int, MonitorElement*> > PEDVALS;
    std::map<HcalCastorDetId,std::map<int, MonitorElement*> > SUBVALS;

    MonitorElement* ALLPEDS;
    MonitorElement* PEDRMS;

    ////---- data from the Condition Database
    MonitorElement* PEDESTAL_REFS;
    MonitorElement* WIDTH_REFS;

  } castHists ;

};

#endif

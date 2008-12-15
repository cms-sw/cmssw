#ifndef DQM_CASTORMONITOR_CASTORPEDESTALMONITOR_H
#define DQM_CASTORMONITOR_CASTORPEDESTALMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"

#include "CondFormats/CastorObjects/interface/CastorPedestal.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidth.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class CastorPedestalMonitor: public CastorBaseMonitor {
public:
  CastorPedestalMonitor(); 
  ~CastorPedestalMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);

  void processEvent(const CastorDigiCollection& cast,const CastorDbService& cond);


  void done();
  void reset();

private: 
  void perChanHists(vector<HcalCastorDetId> detID, vector<int> capID, vector<float> peds,
		    map<HcalCastorDetId, map<int, MonitorElement*> > &toolP, 
		    ////// map<HcalCastorDetId, map<int, MonitorElement*> > &toolS,
		    string baseFolder);

  bool doPerChannel_;
  bool doFCpeds_;
  map<HcalCastorDetId, map<int,MonitorElement*> >::iterator meo_;
  vector<HcalCastorDetId> detID_;
  vector<int> capID_;
  vector<float> pedVals_;

  string outputFile_;
   
  const CastorQIEShape* shape_;
  const CastorQIECoder* channelCoder_;
  CastorCalibrations calibs_; 
  

  MonitorElement* meEVT_;
  int ievt_;
    
  map<HcalCastorDetId,bool> REG;
  MonitorElement* PEDESTAL_REFS;
  MonitorElement* WIDTH_REFS;

  struct{
    map<HcalCastorDetId,map<int, MonitorElement*> > PEDVALS;
    map<HcalCastorDetId,map<int, MonitorElement*> > SUBVALS;

    MonitorElement* ALLPEDS;
    MonitorElement* PEDRMS;

    ////---- data from the Condition Database
    MonitorElement* PEDESTAL_REFS;
    MonitorElement* WIDTH_REFS;

  } castHists ;

};

#endif

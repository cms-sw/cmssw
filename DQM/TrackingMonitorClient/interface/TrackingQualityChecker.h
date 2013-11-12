#ifndef _TrackingQualityChecker_h_
#define _TrackingQualityChecker_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

class DQMStore;
class MonitorElement;
class TkDetMap;
class TrackingDetCabling;

class TrackingQualityChecker {

 public:


  TrackingQualityChecker(edm::ParameterSet const& ps);
  virtual ~TrackingQualityChecker();


  void bookGlobalStatus(DQMStore* dqm_store);     
  void bookLSStatus(DQMStore* dqm_store);     
  void resetGlobalStatus();
  void resetLSStatus();
  void fillDummyGlobalStatus();
  void fillDummyLSStatus();
  void fillGlobalStatus(DQMStore* dqm_store);
  void fillLSStatus(DQMStore* dqm_store);
  
 private:

  struct TrackingMEs{
    MonitorElement* TrackingFlag;
    std::string     HistoDir;
    std::string     HistoName;
  };

  struct TrackingLSMEs{
    MonitorElement* TrackingFlag;
    std::string     HistoLSDir;
    std::string     HistoLSName;
    float           HistoLSLowerCut;
    float           HistoLSUpperCut; 
  };

  void fillTrackingStatus(DQMStore* dqm_store); 
  void fillTrackingStatusAtLumi(DQMStore* dqm_store);

  void fillStatusHistogram(MonitorElement*, int xbin, int ybin, float val);

  
  std::map<std::string, TrackingMEs>   TrackingMEsMap;
  std::map<std::string, TrackingLSMEs> TrackingLSMEsMap;
  
  MonitorElement* TrackGlobalSummaryReportMap;
  MonitorElement* TrackGlobalSummaryReportGlobal;

  MonitorElement* TrackLSSummaryReportGlobal;

  edm::ParameterSet pSet_;

  bool bookedTrackingGlobalStatus_;
  bool bookedTrackingLSStatus_;

  std::string TopFolderName_;

};
#endif

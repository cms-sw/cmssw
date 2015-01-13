#ifndef _SiStripQualityChecker_h_
#define _SiStripQualityChecker_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

class MonitorElement;
class TkDetMap;
class SiStripDetCabling;

class SiStripQualityChecker {

 public:


  SiStripQualityChecker(edm::ParameterSet const& ps);
  virtual ~SiStripQualityChecker();


  void bookStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);     
  void resetStatus();
  void fillDummyStatus();
  void fillStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const edm::ESHandle< SiStripDetCabling >& cabling, const TrackerTopology *tTopo);
  void fillStatusAtLumi(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
  void printStatusReport();
  void fillFaultyModuleStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const TrackerTopology *tTopo);
  
 private:

  struct SubDetMEs{
    MonitorElement* DetFraction;
    MonitorElement* SToNFlag;
    MonitorElement* SummaryFlag;
    std::string     detectorTag;
  };

  void fillDetectorStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const edm::ESHandle< SiStripDetCabling >& cabling);
  void fillSubDetStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,const edm::ESHandle< SiStripDetCabling >& cabling, SubDetMEs& mes, unsigned int xbin,float& gflag);
  void getModuleStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::vector<MonitorElement*>& layer_mes, int& errdet);

  void fillStatusHistogram(MonitorElement*, int xbin, int ybin, float val);
  void initialiseBadModuleList();  

  void fillDetectorStatusAtLumi(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
  
  std::map<std::string, SubDetMEs> SubDetMEsMap;
  std::map<std::string, std::string> SubDetFolderMap;
  
  MonitorElement* DetFractionReportMap;
  MonitorElement* SToNReportMap;
  MonitorElement* SummaryReportMap;

  MonitorElement* SummaryReportGlobal;

  MonitorElement* TrackSummaryReportMap;

  MonitorElement* TrackSummaryReportGlobal;

  std::map<uint32_t,uint16_t> badModuleList;
 
  edm::ParameterSet pSet_;

  bool bookedStripStatus_;
  int globalStatusFilling_;
  bool useGoodTracks_;

  TkDetMap* tkDetMap_;
 
  float cutoffTrackRate_;
  float cutoffChi2overDoF_;
  float cutoffRecHits_;

};
#endif

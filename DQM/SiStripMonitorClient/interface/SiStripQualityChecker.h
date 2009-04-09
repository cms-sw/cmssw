#ifndef _SiStripQualityChecker_h_
#define _SiStripQualityChecker_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

class DQMStore;
class MonitorElement;

class SiStripQualityChecker {

 public:

  SiStripQualityChecker(edm::ParameterSet const& ps);
  virtual ~SiStripQualityChecker();


  void bookStatus(DQMStore* dqm_store);     
  void resetStatus();
  void fillDummyStatus();
  void fillStatus(DQMStore* dqm_store);
  void printStatusReport();

 private:


  struct SubDetMEs{
    MonitorElement* DetFraction;
    MonitorElement* SToNFlag;
    MonitorElement* SummaryFlag;
  };

  void fillSubDetStatus(DQMStore* dqm_store,SubDetMEs& mes,
                                 unsigned int xbin,float& gflag);
  void getModuleStatus(DQMStore* dqm_store,int& ndet,int& errdet);
  void getModuleStatus(MonitorElement* me, int& ndet, std::vector<DQMChannel>& bad_channels);

  void fillStatusHistogram(MonitorElement*, int xbin, int ybin, float val);

  std::map<std::string, SubDetMEs> SubDetMEsMap;
  std::map<std::string, std::string> SubDetFolderMap;

  
  MonitorElement* DetFractionReportMap;
  MonitorElement* SToNReportMap;
  MonitorElement* SummaryReportMap;

  MonitorElement* SummaryReportGlobal;

  edm::ParameterSet pSet_;
  bool bookedStatus_;
  int globalStatusFilling_;
};
#endif

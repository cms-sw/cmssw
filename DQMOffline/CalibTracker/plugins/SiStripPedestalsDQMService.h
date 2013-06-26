#ifndef DQMOffline_SiStripPedestalsDQMService_SiStripPedestalsDQMService_H
#define DQMOffline_SiStripPedestalsDQMService_SiStripPedestalsDQMService_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "DQMOffline/CalibTracker/interface/SiStripBaseServiceFromDQM.h"
#include <string>
#include <memory>
#include <sstream>
// #include "DQM/SiStripMonitorClient/interface/SiStripQualityChecker.h"



/**
  @class SiStripPedestalsDQMService
  @author M. De Mattia, S. Dutta, D. Giordano
  @EDAnalyzer to read modules flagged by the DQM as bad and write in the database.
*/

// class SiStripPedestalsDQMService : public SiStripCondObjBuilderBase<SiStripBadStrip>, public SiStripBaseServiceFromDQM<SiStripBadStrip>
class SiStripPedestalsDQMService : public SiStripBaseServiceFromDQM<SiStripPedestals>
{
 public:

  explicit SiStripPedestalsDQMService(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripPedestalsDQMService();
  
  void getObj(SiStripPedestals* & obj)
  {
    readPedestals(); obj=obj_;
  }

 private:

  void readPedestals();

  edm::ParameterSet iConfig_;
  edm::FileInPath fp_;
};

#endif //DQMOffline_SiStripPedestalsDQMService_SiStripPedestalsDQMService_H

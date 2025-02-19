#ifndef DQMOffline_SiStripNoisesDQMService_SiStripNoisesDQMService_H
#define DQMOffline_SiStripNoisesDQMService_SiStripNoisesDQMService_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "DQMOffline/CalibTracker/interface/SiStripBaseServiceFromDQM.h"
#include <string>
#include <memory>
#include <sstream>
// #include "DQM/SiStripMonitorClient/interface/SiStripQualityChecker.h"



/**
  @class SiStripNoisesDQMService
  @author M. De Mattia, S. Dutta, D. Giordano
  @EDAnalyzer to read modules flagged by the DQM as bad and write in the database.
*/

// class SiStripNoisesDQMService : public SiStripCondObjBuilderBase<SiStripBadStrip>, public SiStripBaseServiceFromDQM<SiStripBadStrip>
class SiStripNoisesDQMService : public SiStripBaseServiceFromDQM<SiStripNoises>
{
 public:

  explicit SiStripNoisesDQMService(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripNoisesDQMService();
  
  void getObj(SiStripNoises* & obj)
  {
    readNoises(); obj=obj_;
  }

 private:

  void readNoises();

  edm::ParameterSet iConfig_;
  edm::FileInPath fp_;
};

#endif //DQMOffline_SiStripNoisesDQMService_SiStripNoisesDQMService_H

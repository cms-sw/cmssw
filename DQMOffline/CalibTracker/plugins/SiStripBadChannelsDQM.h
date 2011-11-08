#ifndef DQMOffline_CalibTracker_SiStripBadChannelsDQM_H
#define DQMOffline_CalibTracker_SiStripBadChannelsDQM_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMOffline/CalibTracker/interface/SiStripBaseServiceFromDQM.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include <TFile.h>
#include <string>
#include <map>

class SiStripBadChannelsDQM : public edm::EDAnalyzer, public SiStripBaseServiceFromDQM<SiStripBadStrip>
{
 public:
  SiStripBadChannelsDQM(const edm::ParameterSet& iConfig);
  ~SiStripBadChannelsDQM();
  
 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  bool readBadChannels();

  void addBadStrips(const unsigned int aDetId,
		    const unsigned short aApvNum,
		    const unsigned short aFlag);
  
  /// Writes the errors to the db
  void addErrors();

  edm::ParameterSet iConfig_;
  edm::FileInPath fp_;
  
  TkDetMap* tkDetMap_;

  double threshold_;
  unsigned int debug_;
  std::map<uint32_t, std::vector<unsigned int> > detIdErrors_;
};

#endif

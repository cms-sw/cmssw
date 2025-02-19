#ifndef DQMOffline_CalibTracker_SiStripFEDErrorsDQM_H
#define DQMOffline_CalibTracker_SiStripFEDErrorsDQM_H

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

#include <TFile.h>
#include <string>
#include <map>

/**
  @class SiStripFEDErrorsDQM
  @author A.-M. Magnan, M. De Mattia
  @EDAnalyzer to read modules flagged by the DQM due to FED errors as bad and write in the database with the proper error flag.
*/

class SiStripFEDErrorsDQM : public edm::EDAnalyzer, public SiStripBaseServiceFromDQM<SiStripBadStrip>
{
 public:
  SiStripFEDErrorsDQM(const edm::ParameterSet& iConfig);
  ~SiStripFEDErrorsDQM();
  
 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  bool readBadAPVs();

  void readHistogram(MonitorElement* aMe,
		     unsigned int & aCounter, 
		     const float aNorm,
		     const unsigned int aFedId);

  void addBadAPV(const FedChannelConnection & aConnection,
		 const unsigned short aAPVNumber,
		 const unsigned short aFlag,
		 unsigned int & aCounter);

  void addBadStrips(const FedChannelConnection & aConnection,
		    const unsigned int aDetId,
		    const unsigned short aApvNum,
		    const unsigned short aFlag,
		    unsigned int & aCounter);

  /// Writes the errors to the db
  void addErrors();

  //set corresponding bit to 1 in flag
  void setFlagBit(unsigned short & aFlag, const unsigned short aBit);

  //update the cabling if necessary
  void updateCabling(const edm::EventSetup& eventSetup);

  edm::ParameterSet iConfig_;
  edm::FileInPath fp_;

  uint32_t cablingCacheId_;
  const SiStripFedCabling* cabling_;

  double threshold_;
  unsigned int debug_;
  std::map<uint32_t, std::vector<unsigned int> > detIdErrors_;
};

#endif

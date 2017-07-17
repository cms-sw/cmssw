#ifndef _DQMOFFLINE_HCAL_CALOTOWERSDQMCLIENT_H_
#define _DQMOFFLINE_HCAL_CALOTOWERSDQMCLIENT_H_

// -*- C++ -*-
//
// 
/*
 Description: This is a CaloTowers client meant to plot calotowers quantities 
*/

//
// Originally create by: Hongxuan Liu 
//                        May 2010
//

#include <memory>
#include <unistd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <iostream>
#include <fstream>
#include <vector>

class DQMStore;
class MonitorElement;

class CaloTowersDQMClient : public DQMEDHarvester {
 
 private:
  std::string outputFile_;
  edm::ParameterSet conf_;

  bool verbose_;
  bool debug_;

  std::string dirName_;
  std::string dirNameJet_;
  std::string dirNameMET_;

 public:
  explicit CaloTowersDQMClient(const edm::ParameterSet& );
  virtual ~CaloTowersDQMClient();
  
  virtual void beginJob(void) override;
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c) override;

  int CaloTowersEndjob(const std::vector<MonitorElement*> &hcalMEs);

};
 
#endif

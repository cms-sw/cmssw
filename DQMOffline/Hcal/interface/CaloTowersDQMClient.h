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

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

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
  explicit CaloTowersDQMClient(const edm::ParameterSet &);
  ~CaloTowersDQMClient() override;

  void beginJob(void) override;
  void dqmEndJob(DQMStore::IBooker &,
                 DQMStore::IGetter &) override;  // performed in the endJob
  void beginRun(const edm::Run &run, const edm::EventSetup &c) override;

  int CaloTowersEndjob(const std::vector<MonitorElement *> &hcalMEs);
};

#endif

#ifndef _DQMOFFLINE_HCAL_HCALNOISERATESCLIENT_H_
#define _DQMOFFLINE_HCAL_HCALNOISERATESCLIENT_H_

// -*- C++ -*-
//
//
/*
 Description: This is a NoiseRates client meant to plot noiserates quantities
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

class HcalNoiseRatesClient : public DQMEDHarvester {
private:
  std::string outputFile_;

  edm::ParameterSet conf_;

  bool verbose_;
  bool debug_;

  std::string dirName_;
  std::string dirNameJet_;
  std::string dirNameMET_;

public:
  explicit HcalNoiseRatesClient(const edm::ParameterSet &);
  ~HcalNoiseRatesClient() override;

  void beginJob(void) override;
  void dqmEndJob(DQMStore::IBooker &,
                 DQMStore::IGetter &) override;  // performed in the endJob
  void beginRun(const edm::Run &run, const edm::EventSetup &c) override;

  int NoiseRatesEndjob(const std::vector<MonitorElement *> &hcalMEs);
};

#endif

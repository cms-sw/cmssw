#ifndef SiStripMonitorCluster_SiStripMonitorFilter_h
#define SiStripMonitorCluster_SiStripMonitorFilter_h
// -*- C++ -*-
//
// Package:     SiStripMonitorCluster
// Class  :     SiStripMonitorFilter
// Original Author: dkcira

// system include files
#include <memory>

// user include files
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DQMServices/Core/interface/DQMStore.h"


class SiStripMonitorFilter : public DQMEDAnalyzer {
public:
  explicit SiStripMonitorFilter(const edm::ParameterSet &);
  ~SiStripMonitorFilter() override{};

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  edm::EDGetTokenT<int> filerDecisionToken_;
  edm::ParameterSet conf_;
  MonitorElement *FilterDecision;
  // all events
  std::string FilterDirectory;
};

#endif

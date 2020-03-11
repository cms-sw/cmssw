#ifndef SiStripMonitorClient_SiStripOfflineDQM_h
#define SiStripMonitorClient_SiStripOfflineDQM_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripOfflineDQM
//
/**\class SiStripOfflineDQM SiStripOfflineDQM.h
   DQM/SiStripMonitorCluster/interface/SiStripOfflineDQM.h

   Description:
   DQM class to perform Summary creation Quality Test on a merged Root file
   after CAF processing
   Usage:
   <usage>

*/
//
// Original Author:  Samvel Khalatyan (ksamdev at gmail dot com)
//         Created:  Wed Oct 5 16:47:14 CET 2006
//

#include <string>

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <TTree.h>

class SiStripDetCabling;

class SiStripOfflineDQM : public edm::one::EDProducer<edm::one::WatchLuminosityBlocks,
                                                      edm::one::WatchRuns,
                                                      edm::EndRunProducer,
                                                      edm::EndLuminosityBlockProducer,
                                                      edm::one::SharedResources> {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripOfflineDQM(edm::ParameterSet const& ps);

private:
  void beginJob() override;
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) override{};
  void produce(edm::Event& e, edm::EventSetup const& eSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) override;
  void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override{};
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void endRunProduce(edm::Run&, edm::EventSetup const&) override{};
  void endJob() override;

  void checkTrackerFEDs(edm::Event const& e);
  bool openInputFile(DQMStore& dqm_store);

  edm::ParameterSet const configPar_;

  SiStripActionExecutor actionExecutor_;

  bool usedWithEDMtoMEConverter_;
  bool createSummary_;
  bool const createTkInfoFile_;
  std::string const inputFileName_;
  std::string const outputFileName_;
  int globalStatusFilling_;
  bool trackerFEDsFound_;
  bool printFaultyModuleList_;
  TTree* tkinfoTree_{nullptr};
};
#endif

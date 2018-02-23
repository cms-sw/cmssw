#ifndef CORE_DQMED_HARVESTER_H
#define CORE_DQMED_HARVESTER_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

class DQMEDHarvester: public edm::one::EDProducer<edm::EndLuminosityBlockProducer,
                                                  edm::EndRunProducer,
                                                  edm::one::WatchLuminosityBlocks,
                                                  edm::one::WatchRuns,
                                                  edm::one::SharedResources>
{

public:
  DQMEDHarvester();
  ~DQMEDHarvester() override = default;

  // implicit copy constructor
  // implicit assignment operator
  // implicit destructor
  void beginRun(edm::Run const&, edm::EventSetup const&) override {};
  void endRun(edm::Run const&, edm::EventSetup const&) override {};
  void endRunProduce(edm::Run& run, edm::EventSetup const& setup) override;

  void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const&) final {};
  void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const&) final;
  void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) final;

  void endJob() final;
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) {};
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) = 0;
};

#endif // CORE_DQMED_HARVESTER_H

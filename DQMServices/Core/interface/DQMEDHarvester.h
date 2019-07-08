#ifndef CORE_DQMED_HARVESTER_H
#define CORE_DQMED_HARVESTER_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

class DQMEDHarvester : public edm::one::EDProducer<edm::Accumulator,
                                                   edm::EndLuminosityBlockProducer,
                                                   edm::EndRunProducer,
                                                   edm::one::WatchLuminosityBlocks,
                                                   edm::one::WatchRuns,
                                                   edm::one::SharedResources> {
public:
  DQMEDHarvester();
  ~DQMEDHarvester() override = default;

  void accumulate(edm::Event const &ev, edm::EventSetup const &es) final{};

  void beginRun(edm::Run const &, edm::EventSetup const &) override{};
  void endRun(edm::Run const &, edm::EventSetup const &) override{};
  void endRunProduce(edm::Run &run, edm::EventSetup const &setup) override;

  void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) final{};
  void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) final;
  void endLuminosityBlockProduce(edm::LuminosityBlock &, edm::EventSetup const &) final;

  void endJob() final;
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &,
                                     DQMStore::IGetter &,
                                     edm::LuminosityBlock const &,
                                     edm::EventSetup const &){};
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) = 0;

protected:
  edm::EDPutTokenT<DQMToken> lumiToken_;
  edm::EDPutTokenT<DQMToken> runToken_;
};

#endif  // CORE_DQMED_HARVESTER_H

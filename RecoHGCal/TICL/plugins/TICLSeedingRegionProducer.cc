// Authors: Felice Pantaleo, Marco Rovere
// Emails: felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 06/2019

#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoHGCal/TICL/plugins/SeedingRegionAlgoBase.h"
#include "SeedingRegionAlgoFactory.h"
#include "SeedingRegionByL1.h"
#include "SeedingRegionByTracks.h"
#include "SeedingRegionGlobal.h"
#include "SeedingRegionByHF.h"

using namespace ticl;

class TICLSeedingRegionProducer : public edm::stream::EDProducer<> {
public:
  TICLSeedingRegionProducer(const edm::ParameterSet&);
  ~TICLSeedingRegionProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginRun(edm::Run const& iEvent, edm::EventSetup const& es) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  std::unique_ptr<SeedingRegionAlgoBase> myAlgo_;
};

DEFINE_FWK_MODULE(TICLSeedingRegionProducer);

TICLSeedingRegionProducer::TICLSeedingRegionProducer(const edm::ParameterSet& ps) {
  auto sumes = consumesCollector();
  auto seedingPSet = ps.getParameter<edm::ParameterSet>("seedingPSet");
  auto algoType = seedingPSet.getParameter<std::string>("type");
  myAlgo_ = SeedingRegionAlgoFactory::get()->create(algoType, seedingPSet, sumes);
  produces<std::vector<TICLSeedingRegion>>();
}

void TICLSeedingRegionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription seedingDesc;
  seedingDesc.addNode(edm::PluginDescription<SeedingRegionAlgoFactory>("type", "SeedingRegionGlobal", true));
  desc.add<edm::ParameterSetDescription>("seedingPSet", seedingDesc);
  descriptions.add("ticlSeedingRegionProducer", desc);
}

void TICLSeedingRegionProducer::beginRun(edm::Run const& iEvent, edm::EventSetup const& es) { myAlgo_->initialize(es); }

void TICLSeedingRegionProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<TICLSeedingRegion>>();
  myAlgo_->makeRegions(evt, es, *result);

  evt.put(std::move(result));
}

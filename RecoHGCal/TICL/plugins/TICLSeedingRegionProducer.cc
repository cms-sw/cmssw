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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoHGCal/TICL/plugins/SeedingRegionAlgoBase.h"
#include "SeedingRegionByTracks.h"
#include "SeedingRegionGlobal.h"

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
  int algoId_;
  std::string seedingId_;
};
DEFINE_FWK_MODULE(TICLSeedingRegionProducer);

TICLSeedingRegionProducer::TICLSeedingRegionProducer(const edm::ParameterSet& ps)
    : algoId_(ps.getParameter<int>("algoId")) {
  auto sumes = consumesCollector();

  switch (algoId_) {
    case 1:
      myAlgo_ = std::make_unique<SeedingRegionByTracks>(ps, sumes);
      break;
    case 2:
      myAlgo_ = std::make_unique<SeedingRegionGlobal>(ps, sumes);
      break;
    default:
      break;
  }
  produces<std::vector<TICLSeedingRegion>>();
}

void TICLSeedingRegionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("algo_verbosity", 0);
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 2. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 10");
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<int>("algoId", 1);
  descriptions.add("ticlSeedingRegionProducer", desc);
}

void TICLSeedingRegionProducer::beginRun(edm::Run const& iEvent, edm::EventSetup const& es) { myAlgo_->initialize(es); }

void TICLSeedingRegionProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<TICLSeedingRegion>>();
  myAlgo_->makeRegions(evt, es, *result);

  evt.put(std::move(result));
}

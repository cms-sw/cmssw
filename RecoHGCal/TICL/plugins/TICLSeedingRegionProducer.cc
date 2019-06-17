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
#include "RecoHGCal/TICL/interface/SeedingRegionAlgoBase.h"
#include "SeedingRegionByTracks.h"

using namespace ticl;

class TICLSeedingRegionProducer : public edm::stream::EDProducer<> {
public:
  TICLSeedingRegionProducer(const edm::ParameterSet&);
  ~TICLSeedingRegionProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:

  std::unique_ptr<SeedingRegionAlgoBase> myAlgo_;
};
DEFINE_FWK_MODULE(TICLSeedingRegionProducer);

TICLSeedingRegionProducer::TICLSeedingRegionProducer(const edm::ParameterSet& ps)   
{
  auto sumes = consumesCollector();
  myAlgo_ = std::make_unique<SeedingRegionByTracks>(ps,sumes);
  produces<std::vector<ticl::TICLSeedingRegion>>();
}

void TICLSeedingRegionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<int>("algo_verbosity", 0);

  descriptions.add("seedingRegionProducer", desc);
}

void TICLSeedingRegionProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<ticl::TICLSeedingRegion>>();
  myAlgo_->makeRegions(evt, es, *result);

  evt.put(std::move(result));
}

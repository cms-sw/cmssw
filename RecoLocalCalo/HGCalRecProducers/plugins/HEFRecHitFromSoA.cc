#include "HEFRecHitFromSoA.h"

HEFRecHitFromSoA::HEFRecHitFromSoA(const edm::ParameterSet& ps) {
  recHitSoAToken_ = consumes<HGCRecHitSoA>(ps.getParameter<edm::InputTag>("HEFRecHitSoATok"));
  recHitCollectionToken_ = produces<HGChefRecHitCollection>(collectionName_);
}

HEFRecHitFromSoA::~HEFRecHitFromSoA() {}

void HEFRecHitFromSoA::acquire(edm::Event const& event,
                               edm::EventSetup const& setup,
                               edm::WaitingTaskWithArenaHolder w) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w)};
  HGCRecHitSoA recHitsSoA = event.get(recHitSoAToken_);

  rechits_ = std::make_unique<HGCRecHitCollection>();

  convert_soa_data_to_collection_(recHitsSoA.nhits_, *rechits_, &recHitsSoA);
}

void HEFRecHitFromSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  event.put(std::move(rechits_), collectionName_);
}

void HEFRecHitFromSoA::convert_soa_data_to_collection_(const uint32_t& nhits,
                                                       HGCRecHitCollection& rechits,
                                                       HGCRecHitSoA* h_calibSoA) {
  rechits.reserve(nhits);
  for (uint i = 0; i < nhits; ++i) {
    DetId id_converted(h_calibSoA->id_[i]);
    rechits.emplace_back(HGCRecHit(id_converted,
                                   h_calibSoA->energy_[i],
                                   h_calibSoA->time_[i],
                                   0,
                                   h_calibSoA->flagBits_[i],
                                   h_calibSoA->son_[i],
                                   h_calibSoA->timeError_[i]));
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HEFRecHitFromSoA);

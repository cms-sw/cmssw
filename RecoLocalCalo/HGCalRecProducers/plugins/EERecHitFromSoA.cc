#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "CUDADataFormats/HGCal/interface/HGCRecHitCPUProduct.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"

class EERecHitFromSoA : public edm::stream::EDProducer<> {
public:
  explicit EERecHitFromSoA(const edm::ParameterSet& ps);
  ~EERecHitFromSoA() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void convert_soa_data_to_collection_(uint32_t, HGCRecHitCollection&, ConstHGCRecHitSoA*);

private:
  std::unique_ptr<HGCeeRecHitCollection> rechits_;
  edm::EDGetTokenT<HGCRecHitCPUProduct> recHitSoAToken_;
  edm::EDPutTokenT<HGCeeRecHitCollection> recHitCollectionToken_;
};

EERecHitFromSoA::EERecHitFromSoA(const edm::ParameterSet& ps) {
  recHitSoAToken_ = consumes<HGCRecHitCPUProduct>(ps.getParameter<edm::InputTag>("EERecHitSoATok"));
  recHitCollectionToken_ = produces<HGCeeRecHitCollection>();
}

EERecHitFromSoA::~EERecHitFromSoA() {}

void EERecHitFromSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  const HGCRecHitCPUProduct& recHits = event.get(recHitSoAToken_);
  ConstHGCRecHitSoA recHitsSoA = recHits.get();
  rechits_ = std::make_unique<HGCRecHitCollection>();
  convert_soa_data_to_collection_(recHits.nHits(), *rechits_, &recHitsSoA);
  event.put(std::move(rechits_));
}

void EERecHitFromSoA::convert_soa_data_to_collection_(uint32_t nhits,
                                                      HGCRecHitCollection& rechits,
                                                      ConstHGCRecHitSoA* h_calibSoA) {
  rechits.reserve(nhits);
  for (uint i = 0; i < nhits; ++i) {
    DetId id_converted(h_calibSoA->id_[i]);
    rechits.emplace_back(id_converted,
                         h_calibSoA->energy_[i],
                         h_calibSoA->time_[i],
                         0,
                         h_calibSoA->flagBits_[i],
                         h_calibSoA->son_[i],
                         h_calibSoA->timeError_[i]);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EERecHitFromSoA);

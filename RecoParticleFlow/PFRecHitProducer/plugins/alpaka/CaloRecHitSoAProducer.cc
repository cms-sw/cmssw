#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "CalorimeterDefinitions.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitHostCollection.h"
#include "DataFormats/HcalRecHit/interface/alpaka/HcalRecHitDeviceCollection.h"

#define DEBUG false

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace particleFlowRecHitProducer;

  template <typename CAL>
  class CaloRecHitSoAProducer : public global::EDProducer<> {
  public:
    CaloRecHitSoAProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          recHitsToken_(consumes(config.getParameter<edm::InputTag>("src"))),
          deviceToken_(produces()),
          synchronise_(config.getUntrackedParameter<bool>("synchronise")) {}

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      const edm::SortedCollection<typename CAL::CaloRecHitType>& recHits = event.get(recHitsToken_);
      const int32_t num_recHits = recHits.size();
      if (DEBUG)
        printf("Found %d recHits\n", num_recHits);

      hcal::RecHitHostCollection hostProduct{num_recHits, event.queue()};
      auto& view = hostProduct.view();

      for (int i = 0; i < num_recHits; i++) {
        convertRecHit(view[i], recHits[i]);

        if (DEBUG && i < 10)
          printf("recHit %4d %u %f %f\n", i, view.detId(i), view.energy(i), view.timeM0(i));
      }

      hcal::RecHitDeviceCollection deviceProduct{num_recHits, event.queue()};
      alpaka::memcpy(event.queue(), deviceProduct.buffer(), hostProduct.buffer());
      if (synchronise_)
        alpaka::wait(event.queue());
      event.emplace(deviceToken_, std::move(deviceProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag(""))->setComment("Input calorimeter rec hit collection");
      desc.addUntracked<bool>("synchronise", false)
          ->setComment("Add synchronisation point after execution (for benchmarking asynchronous execution)");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<edm::SortedCollection<typename CAL::CaloRecHitType>> recHitsToken_;
    const device::EDPutToken<hcal::RecHitDeviceCollection> deviceToken_;
    const bool synchronise_;

    static void convertRecHit(hcal::RecHitHostCollection::View::element to, const typename CAL::CaloRecHitType& from);
  };

  template <>
  void CaloRecHitSoAProducer<HCAL>::convertRecHit(hcal::RecHitHostCollection::View::element to,
                                                  const HCAL::CaloRecHitType& from) {
    // Fill SoA from HCAL rec hit
    to.detId() = from.id().rawId();
    to.energy() = from.energy();
    to.timeM0() = from.time();
  }

  /*
  The ECALRecHitSoAProducer currently has no use, but is available via this
  module. In the case where ECAL PF Clustering is moved to Alpaka, we can then
  decide to use this converted solely for ECAL, or if the SoA is available
  we can switch to using just the ECAL RecHit SoA.
  */

  /*
  template <>
  void CaloRecHitSoAProducer<ECAL>::convertRecHit(reco::CaloRecHitHostCollection::View::element to,
                                                  const ECAL::CaloRecHitType& from) {
    // Fill SoA from ECAL rec hit
    to.detId() = from.id().rawId();
    to.energy() = from.energy();
    to.time() = from.time();
    to.flags() = from.flagsBits();
  }
  */

  using HCALRecHitSoAProducer = CaloRecHitSoAProducer<HCAL>;

  // Purposely commented out; see above.
  //using ECALRecHitSoAProducer = CaloRecHitSoAProducer<ECAL>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HCALRecHitSoAProducer);
//DEFINE_FWK_ALPAKA_MODULE(ECALRecHitSoAProducer);

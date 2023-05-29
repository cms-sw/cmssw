#include <iostream>
#include <string>

#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

class HcalCPURecHitsProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit HcalCPURecHitsProducer(edm::ParameterSet const& ps);
  ~HcalCPURecHitsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  const bool produceSoA_;
  const bool produceLegacy_;

  using IProductType = cms::cuda::Product<hcal::RecHitCollection<calo::common::DevStoragePolicy>>;
  const edm::EDGetTokenT<IProductType> recHitsM0TokenIn_;

  using OProductType = hcal::RecHitCollection<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  const edm::EDPutTokenT<OProductType> recHitsM0TokenOut_;
  const edm::EDPutTokenT<HBHERecHitCollection> recHitsLegacyTokenOut_;

  // to pass from acquire to produce
  OProductType tmpRecHits_;
};

void HcalCPURecHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hbheRecHitProducerGPU"});
  desc.add<std::string>("recHitsM0LabelOut", "");
  desc.add<std::string>("recHitsLegacyLabelOut", "");
  desc.add<bool>("produceSoA", true);
  desc.add<bool>("produceLegacy", true);

  confDesc.addWithDefaultLabel(desc);
}

HcalCPURecHitsProducer::HcalCPURecHitsProducer(const edm::ParameterSet& ps)
    : produceSoA_{ps.getParameter<bool>("produceSoA")},
      produceLegacy_{ps.getParameter<bool>("produceLegacy")},
      recHitsM0TokenIn_{consumes<IProductType>(ps.getParameter<edm::InputTag>("recHitsM0LabelIn"))},
      recHitsM0TokenOut_{produceSoA_ ? produces<OProductType>(ps.getParameter<std::string>("recHitsM0LabelOut"))
                                     : edm::EDPutTokenT<OProductType>{}},  // empty token if disabled
      recHitsLegacyTokenOut_{produceLegacy_
                                 ? produces<HBHERecHitCollection>(ps.getParameter<std::string>("recHitsLegacyLabelOut"))
                                 : edm::EDPutTokenT<HBHERecHitCollection>{}}  // empty token if disabled
{}

void HcalCPURecHitsProducer::acquire(edm::Event const& event,
                                     edm::EventSetup const& setup,
                                     edm::WaitingTaskWithArenaHolder taskHolder) {
  // retrieve data/ctx
  auto const& recHitsProduct = event.get(recHitsM0TokenIn_);
  cms::cuda::ScopedContextAcquire ctx{recHitsProduct, std::move(taskHolder)};
  auto const& recHits = ctx.get(recHitsProduct);

  // resize tmp buffers
  tmpRecHits_.resize(recHits.size);

#ifdef HCAL_MAHI_CPUDEBUG
  std::cout << "num rec Hits = " << recHits.size << std::endl;
#endif

  // do not try to copy the rechits if they are empty
  if (recHits.size == 0) {
    return;
  }

  auto lambdaToTransfer = [&ctx](auto& dest, auto* src) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using src_data_type = typename std::remove_pointer<decltype(src)>::type;
    using type = typename vector_type::value_type;
    static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
    cudaCheck(cudaMemcpyAsync(dest.data(), src, dest.size() * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };

  lambdaToTransfer(tmpRecHits_.energy, recHits.energy.get());
  lambdaToTransfer(tmpRecHits_.chi2, recHits.chi2.get());
  lambdaToTransfer(tmpRecHits_.energyM0, recHits.energyM0.get());
  lambdaToTransfer(tmpRecHits_.timeM0, recHits.timeM0.get());
  lambdaToTransfer(tmpRecHits_.did, recHits.did.get());
}

void HcalCPURecHitsProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  if (produceLegacy_) {
    // populate the legacy collection
    auto recHitsLegacy = std::make_unique<HBHERecHitCollection>();
    // did not set size with ctor as there is no setter for did
    recHitsLegacy->reserve(tmpRecHits_.did.size());
    for (uint32_t i = 0; i < tmpRecHits_.did.size(); i++) {
      // skip bad channels
      if (tmpRecHits_.chi2[i] < 0)
        continue;

      // build a legacy rechit with the computed detid and MAHI energy
      recHitsLegacy->emplace_back(HcalDetId{tmpRecHits_.did[i]},
                                  tmpRecHits_.energy[i],
                                  0  // timeRising
      );
      // update the legacy rechit with the Chi2 and M0 values
      recHitsLegacy->back().setChiSquared(tmpRecHits_.chi2[i]);
      recHitsLegacy->back().setRawEnergy(tmpRecHits_.energyM0[i]);
    }

    // put the legacy collection
    event.put(recHitsLegacyTokenOut_, std::move(recHitsLegacy));
  }

  if (produceSoA_) {
    // put the SoA collection
    event.emplace(recHitsM0TokenOut_, std::move(tmpRecHits_));
  }
  // clear the temporary collection for the next event
  tmpRecHits_.resize(0);
}

DEFINE_FWK_MODULE(HcalCPURecHitsProducer);

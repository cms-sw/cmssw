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
  ~HcalCPURecHitsProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  using IProductType = cms::cuda::Product<hcal::RecHitCollection<calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<IProductType> recHitsM0TokenIn_;
  using OProductType = hcal::RecHitCollection<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  edm::EDPutTokenT<OProductType> recHitsM0TokenOut_;
  edm::EDPutTokenT<HBHERecHitCollection> recHitsLegacyTokenOut_;

  // to pass from acquire to produce
  OProductType tmpRecHits_;
};

void HcalCPURecHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hbheRecHitProducerGPU", "recHitsM0HBHE"});
  desc.add<std::string>("recHitsM0LabelOut", "recHitsM0HBHE");
  desc.add<std::string>("recHitsLegacyLabelOut", "recHitsLegacyHBHE");

  confDesc.addWithDefaultLabel(desc);
}

HcalCPURecHitsProducer::HcalCPURecHitsProducer(const edm::ParameterSet& ps)
    : recHitsM0TokenIn_{consumes<IProductType>(ps.getParameter<edm::InputTag>("recHitsM0LabelIn"))},
      recHitsM0TokenOut_{produces<OProductType>(ps.getParameter<std::string>("recHitsM0LabelOut"))},
      recHitsLegacyTokenOut_{produces<HBHERecHitCollection>(ps.getParameter<std::string>("recHitsLegacyLabelOut"))} {}

HcalCPURecHitsProducer::~HcalCPURecHitsProducer() {}

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
  // populate the legacy collection
  auto recHitsLegacy = std::make_unique<HBHERecHitCollection>();
  // did not set size with ctor as there is no setter for did
  recHitsLegacy->reserve(tmpRecHits_.did.size());
  for (uint32_t i = 0; i < tmpRecHits_.did.size(); i++) {
    recHitsLegacy->emplace_back(HcalDetId{tmpRecHits_.did[i]},
                                tmpRecHits_.energy[i],
                                0  // timeRising
    );

    // update newly pushed guy
    (*recHitsLegacy)[i].setChiSquared(tmpRecHits_.chi2[i]);
    (*recHitsLegacy)[i].setRawEnergy(tmpRecHits_.energyM0[i]);
  }

  // put a legacy format
  event.put(recHitsLegacyTokenOut_, std::move(recHitsLegacy));

  // put a new format
  event.emplace(recHitsM0TokenOut_, std::move(tmpRecHits_));
}

DEFINE_FWK_MODULE(HcalCPURecHitsProducer);

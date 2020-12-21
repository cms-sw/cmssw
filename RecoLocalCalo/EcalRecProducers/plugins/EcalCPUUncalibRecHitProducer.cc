#include <iostream>

// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// algorithm specific

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"

class EcalCPUUncalibRecHitProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalCPUUncalibRecHitProducer(edm::ParameterSet const& ps);
  ~EcalCPUUncalibRecHitProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  using InputProduct = cms::cuda::Product<ecal::UncalibratedRecHit<calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<InputProduct> recHitsInEBToken_, recHitsInEEToken_;
  using OutputProduct = ecal::UncalibratedRecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  edm::EDPutTokenT<OutputProduct> recHitsOutEBToken_, recHitsOutEEToken_;

  OutputProduct recHitsEB_, recHitsEE_;
  bool containsTimingInformation_;
};

void EcalCPUUncalibRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsInLabelEB", edm::InputTag{"ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEB"});
  desc.add<edm::InputTag>("recHitsInLabelEE", edm::InputTag{"ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEE"});
  desc.add<std::string>("recHitsOutLabelEB", "EcalUncalibRecHitsEB");
  desc.add<std::string>("recHitsOutLabelEE", "EcalUncalibRecHitsEE");
  desc.add<bool>("containsTimingInformation", false);

  confDesc.add("ecalCPUUncalibRecHitProducer", desc);
}

EcalCPUUncalibRecHitProducer::EcalCPUUncalibRecHitProducer(const edm::ParameterSet& ps)
    : recHitsInEBToken_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsInLabelEB"))},
      recHitsInEEToken_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsInLabelEE"))},
      recHitsOutEBToken_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsOutLabelEB"))},
      recHitsOutEEToken_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsOutLabelEE"))},
      containsTimingInformation_{ps.getParameter<bool>("containsTimingInformation")} {}

EcalCPUUncalibRecHitProducer::~EcalCPUUncalibRecHitProducer() {}

void EcalCPUUncalibRecHitProducer::acquire(edm::Event const& event,
                                           edm::EventSetup const& setup,
                                           edm::WaitingTaskWithArenaHolder taskHolder) {
  // retrieve data/ctx
  auto const& ebRecHitsProduct = event.get(recHitsInEBToken_);
  auto const& eeRecHitsProduct = event.get(recHitsInEEToken_);
  cms::cuda::ScopedContextAcquire ctx{ebRecHitsProduct, std::move(taskHolder)};
  auto const& ebRecHits = ctx.get(ebRecHitsProduct);
  auto const& eeRecHits = ctx.get(eeRecHitsProduct);

  // resize the output buffers
  recHitsEB_.resize(ebRecHits.size);
  recHitsEE_.resize(eeRecHits.size);

  auto lambdaToTransfer = [&ctx](auto& dest, auto* src) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using type = typename vector_type::value_type;
    using src_type = typename std::remove_pointer<decltype(src)>::type;
    static_assert(std::is_same<src_type, type>::value && "dst and src data types do not match");
    cudaCheck(cudaMemcpyAsync(dest.data(), src, dest.size() * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };

  // enqeue transfers
  lambdaToTransfer(recHitsEB_.did, ebRecHits.did.get());
  lambdaToTransfer(recHitsEE_.did, eeRecHits.did.get());

  lambdaToTransfer(recHitsEB_.amplitudesAll, ebRecHits.amplitudesAll.get());
  lambdaToTransfer(recHitsEE_.amplitudesAll, eeRecHits.amplitudesAll.get());

  lambdaToTransfer(recHitsEB_.amplitude, ebRecHits.amplitude.get());
  lambdaToTransfer(recHitsEE_.amplitude, eeRecHits.amplitude.get());

  lambdaToTransfer(recHitsEB_.chi2, ebRecHits.chi2.get());
  lambdaToTransfer(recHitsEE_.chi2, eeRecHits.chi2.get());

  lambdaToTransfer(recHitsEB_.pedestal, ebRecHits.pedestal.get());
  lambdaToTransfer(recHitsEE_.pedestal, eeRecHits.pedestal.get());

  lambdaToTransfer(recHitsEB_.flags, ebRecHits.flags.get());
  lambdaToTransfer(recHitsEE_.flags, eeRecHits.flags.get());

  if (containsTimingInformation_) {
    lambdaToTransfer(recHitsEB_.jitter, ebRecHits.jitter.get());
    lambdaToTransfer(recHitsEE_.jitter, eeRecHits.jitter.get());

    lambdaToTransfer(recHitsEB_.jitterError, ebRecHits.jitterError.get());
    lambdaToTransfer(recHitsEE_.jitterError, eeRecHits.jitterError.get());
  }
}

void EcalCPUUncalibRecHitProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  // tmp vectors
  auto recHitsOutEB = std::make_unique<OutputProduct>(std::move(recHitsEB_));
  auto recHitsOutEE = std::make_unique<OutputProduct>(std::move(recHitsEE_));

  // put into event
  event.put(recHitsOutEBToken_, std::move(recHitsOutEB));
  event.put(recHitsOutEEToken_, std::move(recHitsOutEE));
}

DEFINE_FWK_MODULE(EcalCPUUncalibRecHitProducer);

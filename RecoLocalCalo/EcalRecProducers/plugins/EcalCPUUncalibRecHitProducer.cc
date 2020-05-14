#include <iostream>

// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEvent.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// algorithm specific

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit_soa.h"

class EcalCPUUncalibRecHitProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalCPUUncalibRecHitProducer(edm::ParameterSet const& ps);
  ~EcalCPUUncalibRecHitProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>> recHitsInEBToken_, recHitsInEEToken_;
  edm::EDPutTokenT<ecal::UncalibratedRecHit<ecal::Tag::soa>> recHitsOutEBToken_, recHitsOutEEToken_;

  ecal::UncalibratedRecHit<ecal::Tag::soa> recHitsEB_, recHitsEE_;
  bool containsTimingInformation_;
};

void EcalCPUUncalibRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsInLabelEB", edm::InputTag{"ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEB"});
  desc.add<edm::InputTag>("recHitsInLabelEE", edm::InputTag{"ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEE"});
  desc.add<std::string>("recHitsOutLabelEB", "EcalUncalibRecHitsEB");
  desc.add<std::string>("recHitsOutLabelEE", "EcalUncalibRecHitsEE");
  desc.add<bool>("containsTimingInformation", false);

  std::string label = "ecalCPUUncalibRecHitProducer";
  confDesc.add(label, desc);
}

EcalCPUUncalibRecHitProducer::EcalCPUUncalibRecHitProducer(const edm::ParameterSet& ps)
    : recHitsInEBToken_{consumes<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>>(
          ps.getParameter<edm::InputTag>("recHitsInLabelEB"))},
      recHitsInEEToken_{consumes<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>>(
          ps.getParameter<edm::InputTag>("recHitsInLabelEE"))},
      recHitsOutEBToken_{
          produces<ecal::UncalibratedRecHit<ecal::Tag::soa>>(ps.getParameter<std::string>("recHitsOutLabelEB"))},
      recHitsOutEEToken_{
          produces<ecal::UncalibratedRecHit<ecal::Tag::soa>>(ps.getParameter<std::string>("recHitsOutLabelEE"))},
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
    cudaCheck(cudaMemcpyAsync(dest.data(), src, dest.size() * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };

  // enqeue transfers
  lambdaToTransfer(recHitsEB_.did, ebRecHits.did);
  lambdaToTransfer(recHitsEE_.did, eeRecHits.did);

  lambdaToTransfer(recHitsEB_.amplitudesAll, ebRecHits.amplitudesAll);
  lambdaToTransfer(recHitsEE_.amplitudesAll, eeRecHits.amplitudesAll);

  lambdaToTransfer(recHitsEB_.amplitude, ebRecHits.amplitude);
  lambdaToTransfer(recHitsEE_.amplitude, eeRecHits.amplitude);

  lambdaToTransfer(recHitsEB_.chi2, ebRecHits.chi2);
  lambdaToTransfer(recHitsEE_.chi2, eeRecHits.chi2);

  lambdaToTransfer(recHitsEB_.pedestal, ebRecHits.pedestal);
  lambdaToTransfer(recHitsEE_.pedestal, eeRecHits.pedestal);

  lambdaToTransfer(recHitsEB_.flags, ebRecHits.flags);
  lambdaToTransfer(recHitsEE_.flags, eeRecHits.flags);

  if (containsTimingInformation_) {
    lambdaToTransfer(recHitsEB_.jitter, ebRecHits.jitter);
    lambdaToTransfer(recHitsEE_.jitter, eeRecHits.jitter);

    lambdaToTransfer(recHitsEB_.jitterError, ebRecHits.jitterError);
    lambdaToTransfer(recHitsEE_.jitterError, eeRecHits.jitterError);
  }
}

void EcalCPUUncalibRecHitProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  // tmp vectors
  auto recHitsOutEB = std::make_unique<ecal::UncalibratedRecHit<ecal::Tag::soa>>(std::move(recHitsEB_));
  auto recHitsOutEE = std::make_unique<ecal::UncalibratedRecHit<ecal::Tag::soa>>(std::move(recHitsEE_));

  // put into event
  event.put(recHitsOutEBToken_, std::move(recHitsOutEB));
  event.put(recHitsOutEEToken_, std::move(recHitsOutEE));
}

DEFINE_FWK_MODULE(EcalCPUUncalibRecHitProducer);

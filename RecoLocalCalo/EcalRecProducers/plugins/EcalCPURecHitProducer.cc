//#define ECAL_RECO_CUDA_DEBUG

#ifdef ECAL_RECO_CUDA_DEBUG
#include <iostream>
#endif

// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

// algorithm specific

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit.h"

class EcalCPURecHitProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalCPURecHitProducer(edm::ParameterSet const& ps);
  ~EcalCPURecHitProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  using InputProduct = cms::cuda::Product<ecal::RecHit<calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<InputProduct> recHitsInEBToken_, recHitsInEEToken_;
  using OutputProduct = ecal::RecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  edm::EDPutTokenT<OutputProduct> recHitsOutEBToken_, recHitsOutEEToken_;

  OutputProduct recHitsEB_, recHitsEE_;
  bool containsTimingInformation_;
};

void EcalCPURecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsInLabelEB", edm::InputTag{"ecalRecHitProducerGPU", "EcalRecHitsEB"});
  desc.add<edm::InputTag>("recHitsInLabelEE", edm::InputTag{"ecalRecHitProducerGPU", "EcalRecHitsEE"});
  desc.add<std::string>("recHitsOutLabelEB", "EcalRecHitsEB");
  desc.add<std::string>("recHitsOutLabelEE", "EcalRecHitsEE");
  desc.add<bool>("containsTimingInformation", false);

  confDesc.addWithDefaultLabel(desc);
}

EcalCPURecHitProducer::EcalCPURecHitProducer(const edm::ParameterSet& ps)
    : recHitsInEBToken_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsInLabelEB"))},
      recHitsInEEToken_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsInLabelEE"))},
      recHitsOutEBToken_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsOutLabelEB"))},
      recHitsOutEEToken_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsOutLabelEE"))},
      containsTimingInformation_{ps.getParameter<bool>("containsTimingInformation")} {}

void EcalCPURecHitProducer::acquire(edm::Event const& event,
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

#ifdef ECAL_RECO_CUDA_DEBUG
  std::cout << " [EcalCPURecHitProducer::acquire] ebRecHits.size = " << ebRecHits.size << std::endl;
  std::cout << " [EcalCPURecHitProducer::acquire] eeRecHits.size = " << eeRecHits.size << std::endl;
#endif

  // enqeue transfers
  cudaCheck(cudaMemcpyAsync(recHitsEB_.did.data(),
                            ebRecHits.did.get(),
                            recHitsEB_.did.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));
  cudaCheck(cudaMemcpyAsync(recHitsEE_.did.data(),
                            eeRecHits.did.get(),
                            recHitsEE_.did.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));
  //
  //     ./CUDADataFormats/EcalRecHitSoA/interface/RecoTypes.h:using StorageScalarType = float;
  //

  cudaCheck(cudaMemcpyAsync(recHitsEB_.energy.data(),
                            ebRecHits.energy.get(),
                            recHitsEB_.energy.size() * sizeof(::ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));
  cudaCheck(cudaMemcpyAsync(recHitsEE_.energy.data(),
                            eeRecHits.energy.get(),
                            recHitsEE_.energy.size() * sizeof(::ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));

  cudaCheck(cudaMemcpyAsync(recHitsEB_.chi2.data(),
                            ebRecHits.chi2.get(),
                            recHitsEB_.chi2.size() * sizeof(::ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));
  cudaCheck(cudaMemcpyAsync(recHitsEE_.chi2.data(),
                            eeRecHits.chi2.get(),
                            recHitsEE_.chi2.size() * sizeof(::ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));

  cudaCheck(cudaMemcpyAsync(recHitsEB_.extra.data(),
                            ebRecHits.extra.get(),
                            recHitsEB_.extra.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));
  cudaCheck(cudaMemcpyAsync(recHitsEE_.extra.data(),
                            eeRecHits.extra.get(),
                            recHitsEE_.extra.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));

  cudaCheck(cudaMemcpyAsync(recHitsEB_.flagBits.data(),
                            ebRecHits.flagBits.get(),
                            recHitsEB_.flagBits.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));
  cudaCheck(cudaMemcpyAsync(recHitsEE_.flagBits.data(),
                            eeRecHits.flagBits.get(),
                            recHitsEE_.flagBits.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));

#ifdef ECAL_RECO_CUDA_DEBUG
  for (unsigned int ieb = 0; ieb < ebRecHits.size; ieb++) {
    if (recHitsEB_.extra[ieb] != 0)
      std::cout << " [ " << ieb << " :: " << ebRecHits.size << " ] [ " << recHitsEB_.did[ieb]
                << " ] eb extra = " << recHitsEB_.extra[ieb] << std::endl;
  }

  for (unsigned int ieb = 0; ieb < ebRecHits.size; ieb++) {
    if (recHitsEB_.energy[ieb] != 0)
      std::cout << " [ " << ieb << " :: " << ebRecHits.size << " ] [ " << recHitsEB_.did[ieb]
                << " ] eb energy = " << recHitsEB_.energy[ieb] << std::endl;
  }

  for (unsigned int iee = 0; iee < eeRecHits.size; iee++) {
    if (recHitsEE_.energy[iee] != 0)
      std::cout << " [ " << iee << " :: " << eeRecHits.size << " ] [ " << recHitsEE_.did[iee]
                << " ] ee energy = " << recHitsEE_.energy[iee] << std::endl;
  }
#endif
}

void EcalCPURecHitProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  // put into event
  event.emplace(recHitsOutEBToken_, std::move(recHitsEB_));
  event.emplace(recHitsOutEEToken_, std::move(recHitsEE_));
}

DEFINE_FWK_MODULE(EcalCPURecHitProducer);

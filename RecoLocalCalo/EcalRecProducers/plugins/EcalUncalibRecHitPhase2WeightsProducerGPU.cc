#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

#include "EcalUncalibRecHitPhase2WeightsAlgoGPU.h"
#include "DeclsForKernelsPhase2.h"

class EcalUncalibRecHitPhase2WeightsProducerGPU : public edm::stream::EDProducer<> {
public:
  explicit EcalUncalibRecHitPhase2WeightsProducerGPU(edm::ParameterSet const &ps);
  ~EcalUncalibRecHitPhase2WeightsProducerGPU() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void produce(edm::Event &, edm::EventSetup const &) override;

private:
  const std::vector<double, cms::cuda::HostAllocator<double>> weights_;

  using InputProduct = cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>;
  const edm::EDGetTokenT<InputProduct> digisToken_;
  using OutputProduct = cms::cuda::Product<ecal::UncalibratedRecHit<calo::common::DevStoragePolicy>>;
  const edm::EDPutTokenT<OutputProduct> recHitsToken_;

  // event data
  ecal::weights::EventOutputDataGPU eventOutputDataGPU_;
};

// constructor with initialisation of elements
EcalUncalibRecHitPhase2WeightsProducerGPU::EcalUncalibRecHitPhase2WeightsProducerGPU(const edm::ParameterSet &ps)
    :  // use lambda to initialise the vector with CUDA::HostAllocator from a normal vector
      weights_([tmp = ps.getParameter<std::vector<double>>("weights")] {
        return std::vector<double, cms::cuda::HostAllocator<double>>(tmp.begin(), tmp.end());
      }()),
      digisToken_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("digisLabelEB"))},
      recHitsToken_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsLabelEB"))} {}

void EcalUncalibRecHitPhase2WeightsProducerGPU::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("recHitsLabelEB", "EcalUncalibRecHitsEB");
  //The below weights values should be kept up to date with those on the CPU version of this module
  desc.add<std::vector<double>>("weights",
                                {-0.121016,
                                 -0.119899,
                                 -0.120923,
                                 -0.0848959,
                                 0.261041,
                                 0.509881,
                                 0.373591,
                                 0.134899,
                                 -0.0233605,
                                 -0.0913195,
                                 -0.112452,
                                 -0.118596,
                                 -0.121737,
                                 -0.121737,
                                 -0.121737,
                                 -0.121737});

  desc.add<edm::InputTag>("digisLabelEB", edm::InputTag("simEcalUnsuppressedDigis", ""));

  descriptions.addWithDefaultLabel(desc);
}

void EcalUncalibRecHitPhase2WeightsProducerGPU::produce(edm::Event &event, const edm::EventSetup &setup) {
  // cuda products
  auto const &digisProduct = event.get(digisToken_);
  // raii
  cms::cuda::ScopedContextProduce ctx{digisProduct};

  // get actual obj
  auto const &digis = ctx.get(digisProduct);

  const uint32_t size = digis.size;

  // do not run the algo if there are no digis
  if (size > 0) {
    auto weights_d = cms::cuda::make_device_unique<double[]>(EcalDataFrame_Ph2::MAXSAMPLES, ctx.stream());

    cudaCheck(cudaMemcpyAsync(weights_d.get(),
                              weights_.data(),
                              EcalDataFrame_Ph2::MAXSAMPLES * sizeof(double),
                              cudaMemcpyHostToDevice,
                              ctx.stream()));

    // output on GPU
    eventOutputDataGPU_.allocate(size, ctx.stream());

    ecal::weights::phase2Weights(digis, eventOutputDataGPU_, weights_d, ctx.stream());
  }

  // set the size of digis
  eventOutputDataGPU_.recHits.size = size;

  // put into the event
  ctx.emplace(event, recHitsToken_, std::move(eventOutputDataGPU_.recHits));
}

DEFINE_FWK_MODULE(EcalUncalibRecHitPhase2WeightsProducerGPU);

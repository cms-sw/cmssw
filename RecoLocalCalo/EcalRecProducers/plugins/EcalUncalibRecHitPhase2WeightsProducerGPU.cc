#include <chrono>

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatiosGPU.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

#include "EcalUncalibRecHitPhase2WeightsAlgoGPU.h"
#include "DeclsForKernelsPh2WeightsGPU.h"


class EcalUncalibRecHitPhase2WeightsProducerGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalUncalibRecHitPhase2WeightsProducerGPU(edm::ParameterSet const &ps);
  ~EcalUncalibRecHitPhase2WeightsProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);


private:
  void acquire(edm::Event const &, edm::EventSetup const &, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event &, edm::EventSetup const &) override;

private:
  const float tRise_;
  const float tFall_;
  const std::vector<double> weights_;

  using InputProduct = cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>;
  const edm::EDGetTokenT<InputProduct> digisTokenEB_;
  using OutputProduct = cms::cuda::Product<ecal::UncalibratedRecHit<calo::common::DevStoragePolicy>>;
  const edm::EDPutTokenT<OutputProduct> recHitsTokenEB_;

  // const edm::ESGetToken<EcalGainRatiosGPU, EcalGainRatiosRcd> gainRatiosToken_;

  // configuration parameters
  // ecal::multifit::ConfigurationParametersWeights configParameters_;

  // event data
  ecal::weights::EventOutputDataGPUWeights eventOutputDataGPU_;

  cms::cuda::ContextState cudaState_;

  uint32_t neb_;

  // timing
  std::chrono::time_point<std::chrono::system_clock> start_, end_;
  std::chrono::duration<float> duration_;
  int event_counter_;
  float ms_[8900];
};

// constructor with initialisation of elements
EcalUncalibRecHitPhase2WeightsProducerGPU::EcalUncalibRecHitPhase2WeightsProducerGPU(const edm::ParameterSet &ps)
    : tRise_(ps.getParameter<double>("tRise")),
      tFall_(ps.getParameter<double>("tFall")),
      weights_(ps.getParameter<std::vector<double>>("weights")),
      digisTokenEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("digisLabelEB"))},
      recHitsTokenEB_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsLabelEB"))} {
        // configParameters_.tRise = tRise_;
        // configParameters_.tFall = tFall_;
        // configParameters_.weights = weights_;

        // timing
        // start_ = std::chrono::high_resolution_clock::now();
        // event_counter_ = 0;
      }

// fill descriptions which describes parameters and has the weights
void EcalUncalibRecHitPhase2WeightsProducerGPU::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("recHitsLabelEB", "EcalUncalibRecHitsEB");
  desc.add<double>("tRise", 0.2);
  desc.add<double>("tFall", 2.);
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

// aquire function which initislises objects on host and device to their actual objects and calls kernal
void EcalUncalibRecHitPhase2WeightsProducerGPU::acquire(edm::Event const &event,
                                           edm::EventSetup const &setup,
                                           edm::WaitingTaskWithArenaHolder holder) {
  // timing

  // end_ = std::chrono::high_resolution_clock::now();
  // duration_ = end_ - start_;
  // float ms = duration_.count()*1000;
  // start_ = std::chrono::high_resolution_clock::now();
  // std::cout<< ms << "ms constructor \n";



  // cuda products
  auto const &ebDigisProduct = event.get(digisTokenEB_);
  // raii
  cms::cuda::ScopedContextAcquire ctx{ebDigisProduct, std::move(holder), cudaState_};

  // get actual obj
  auto const &ebDigis = ctx.get(ebDigisProduct);

  neb_ = ebDigis.size;

  // if no digis stop here
  if (neb_ == 0)
    return;

  // weights to GPU
  
  cms::cuda::device::unique_ptr<double[]> weights_d = cms::cuda::make_device_unique<double[]>(EcalDataFrame_Ph2::MAXSAMPLES, ctx.stream());

  cudaCheck(cudaMemcpyAsync(weights_d.get(),
                            weights_.data(),
                            EcalDataFrame_Ph2::MAXSAMPLES * sizeof(double),
                            cudaMemcpyHostToDevice,
                            ctx.stream()));

  // output on GPU
  eventOutputDataGPU_.allocate(neb_, ctx.stream());

  // cms::cuda::device::unique_ptr<uint16_t[]> debug_d = cms::cuda::make_device_unique<uint16_t[]>(EcalDataFrame_Ph2::MAXSAMPLES, ctx.stream()); //debug
  // std::vector<uint16_t> debug_h;   //debug
  // debug_h.resize(EcalDataFrame_Ph2::MAXSAMPLES);   //debug

  ecal::weights::entryPoint(ebDigis, 
                            eventOutputDataGPU_,
                            weights_d,
                            // debug_d,
                            ctx.stream());

  // timing
  // end_ = std::chrono::high_resolution_clock::now();
  // duration_ = end_ - start_;
  // ms = 0.;
  // ms = duration_.count()*1000;
  // start_ = std::chrono::high_resolution_clock::now();    //kernel timing
  // std::cout<< ms << "ms aquire\n";



  // cudaCheck(cudaMemcpyAsync(debug_h.data(),   //debug
  //                                 debug_d.get(),    //debug
  //                                 EcalDataFrame_Ph2::MAXSAMPLES * sizeof(double),    //debug
  //                                 cudaMemcpyDeviceToHost,    //debug
  //                                 ctx.stream()));    //debug
  // for(int i =0; i<EcalDataFrame_Ph2::MAXSAMPLES; i++){    //debug
  // std::cout << debug_h.data()[i]<<"GPUraw\n";    //debug
  // }   //debug

}

void EcalUncalibRecHitPhase2WeightsProducerGPU::produce(edm::Event& event, const edm::EventSetup& setup) {
  // kernel timing
  //   end_ = std::chrono::high_resolution_clock::now();
  // duration_ = end_ - start_;
  // float ms = 0.;
  // ms = duration_.count()*1000;

  // ms_[event_counter_] = ms;
  // event_counter_ ++;

  // if (event_counter_ == 8900){
  // for (int i =0; i< event_counter_;i++){
  // std::cout<< ms_[i] << "\n";}
  // }

  // start_ = std::chrono::high_resolution_clock::now();


  //DurationMeasurer<std::chrono::milliseconds> timer{std::string{"produce duration"}};
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  // set the size of eb 
  eventOutputDataGPU_.recHitsEB.size = neb_;

  // put into the event
  ctx.emplace(event, recHitsTokenEB_, std::move(eventOutputDataGPU_.recHitsEB));


  // timing
  // end_ = std::chrono::high_resolution_clock::now();
  // duration_ = end_ - start_;
  // ms = duration_.count()*1000;
  // start_ = std::chrono::high_resolution_clock::now();
  // std::cout<< ms << "ms produce\n";


}



DEFINE_FWK_MODULE(EcalUncalibRecHitPhase2WeightsProducerGPU);


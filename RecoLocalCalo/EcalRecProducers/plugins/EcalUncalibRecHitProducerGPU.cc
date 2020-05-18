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
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit_soa.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/Common.h"

#include <iostream>

#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"

#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPedestalsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalGainRatiosGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseShapesGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseCovariancesGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSamplesCorrelationGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalTimeBiasCorrectionsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalTimeCalibConstantsGPU.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/DeclsForKernels.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMultiFitAlgo_gpu_new.h"

class EcalUncalibRecHitProducerGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalUncalibRecHitProducerGPU(edm::ParameterSet const& ps);
  ~EcalUncalibRecHitProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  using RecHitType = ecal::UncalibratedRecHit<ecal::Tag::soa>;
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<cms::cuda::Product<ecal::DigisCollection>> digisTokenEB_, digisTokenEE_;
  edm::EDPutTokenT<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>> recHitsTokenEB_, recHitsTokenEE_;

  // conditions handles
  edm::ESHandle<EcalPedestalsGPU> pedestalsHandle_;
  edm::ESHandle<EcalGainRatiosGPU> gainRatiosHandle_;
  edm::ESHandle<EcalPulseShapesGPU> pulseShapesHandle_;
  edm::ESHandle<EcalPulseCovariancesGPU> pulseCovariancesHandle_;
  edm::ESHandle<EcalSamplesCorrelationGPU> samplesCorrelationHandle_;
  edm::ESHandle<EcalTimeBiasCorrectionsGPU> timeBiasCorrectionsHandle_;
  edm::ESHandle<EcalTimeCalibConstantsGPU> timeCalibConstantsHandle_;
  edm::ESHandle<EcalSampleMask> sampleMaskHandle_;
  edm::ESHandle<EcalTimeOffsetConstant> timeOffsetConstantHandle_;

  // configuration parameters
  ecal::multifit::ConfigurationParameters configParameters_;

  // event data
  ecal::multifit::EventOutputDataGPU eventOutputDataGPU_;
  ecal::multifit::EventDataForScratchGPU eventDataForScratchGPU_;

  cms::cuda::ContextState cudaState_;

  uint32_t maxNumberHits_;
  uint32_t neb_, nee_;
};

void EcalUncalibRecHitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digisLabelEB", edm::InputTag("ecalRawToDigiGPU", "ebDigisGPU"));
  desc.add<edm::InputTag>("digisLabelEE", edm::InputTag("ecalRawToDigiGPU", "eeDigisGPU"));

  desc.add<std::string>("recHitsLabelEB", "EcalUncalibRecHitsEB");
  desc.add<std::string>("recHitsLabelEE", "EcalUncalibRecHitsEE");

  desc.add<std::vector<double>>("EBtimeFitParameters",
                                {-2.015452e+00,
                                 3.130702e+00,
                                 -1.234730e+01,
                                 4.188921e+01,
                                 -8.283944e+01,
                                 9.101147e+01,
                                 -5.035761e+01,
                                 1.105621e+01});
  desc.add<std::vector<double>>("EEtimeFitParameters",
                                {-2.390548e+00,
                                 3.553628e+00,
                                 -1.762341e+01,
                                 6.767538e+01,
                                 -1.332130e+02,
                                 1.407432e+02,
                                 -7.541106e+01,
                                 1.620277e+01});
  desc.add<std::vector<double>>("EBamplitudeFitParameters", {1.138, 1.652});
  desc.add<std::vector<double>>("EEamplitudeFitParameters", {1.890, 1.400});
  desc.add<double>("EBtimeFitLimits_Lower", 0.2);
  desc.add<double>("EBtimeFitLimits_Upper", 1.4);
  desc.add<double>("EEtimeFitLimits_Lower", 0.2);
  desc.add<double>("EEtimeFitLimits_Upper", 1.4);
  desc.add<double>("EBtimeConstantTerm", .6);
  desc.add<double>("EEtimeConstantTerm", 1.0);
  desc.add<double>("EBtimeNconst", 28.5);
  desc.add<double>("EEtimeNconst", 31.8);
  desc.add<double>("outOfTimeThresholdGain12pEB", 5);
  desc.add<double>("outOfTimeThresholdGain12mEB", 5);
  desc.add<double>("outOfTimeThresholdGain61pEB", 5);
  desc.add<double>("outOfTimeThresholdGain61mEB", 5);
  desc.add<double>("outOfTimeThresholdGain12pEE", 1000);
  desc.add<double>("outOfTimeThresholdGain12mEE", 1000);
  desc.add<double>("outOfTimeThresholdGain61pEE", 1000);
  desc.add<double>("outOfTimeThresholdGain61mEE", 1000);
  desc.add<double>("amplitudeThresholdEB", 10);
  desc.add<double>("amplitudeThresholdEE", 10);
  desc.add<uint32_t>("maxNumberHits", 20000);  //---- AM TEST
  desc.add<std::vector<uint32_t>>("kernelMinimizeThreads", {32, 1, 1});
  // ---- default false or true? It was set to true, but at HLT it is false
  desc.add<bool>("shouldRunTimingComputation", false);
  std::string label = "ecalUncalibRecHitProducerGPU";
  confDesc.add(label, desc);
}

EcalUncalibRecHitProducerGPU::EcalUncalibRecHitProducerGPU(const edm::ParameterSet& ps)
    : digisTokenEB_{consumes<cms::cuda::Product<ecal::DigisCollection>>(
          ps.getParameter<edm::InputTag>("digisLabelEB"))},
      digisTokenEE_{
          consumes<cms::cuda::Product<ecal::DigisCollection>>(ps.getParameter<edm::InputTag>("digisLabelEE"))},
      recHitsTokenEB_{produces<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>>(
          ps.getParameter<std::string>("recHitsLabelEB"))},
      recHitsTokenEE_{produces<cms::cuda::Product<ecal::UncalibratedRecHit<ecal::Tag::ptr>>>(
          ps.getParameter<std::string>("recHitsLabelEE"))} {
  auto EBamplitudeFitParameters = ps.getParameter<std::vector<double>>("EBamplitudeFitParameters");
  auto EEamplitudeFitParameters = ps.getParameter<std::vector<double>>("EEamplitudeFitParameters");
  auto EBtimeFitParameters = ps.getParameter<std::vector<double>>("EBtimeFitParameters");
  auto EEtimeFitParameters = ps.getParameter<std::vector<double>>("EEtimeFitParameters");
  std::pair<double, double> EBtimeFitLimits, EEtimeFitLimits;
  EBtimeFitLimits.first = ps.getParameter<double>("EBtimeFitLimits_Lower");
  EBtimeFitLimits.second = ps.getParameter<double>("EBtimeFitLimits_Upper");
  EEtimeFitLimits.first = ps.getParameter<double>("EEtimeFitLimits_Lower");
  EEtimeFitLimits.second = ps.getParameter<double>("EEtimeFitLimits_Upper");

  auto EBtimeConstantTerm = ps.getParameter<double>("EBtimeConstantTerm");
  auto EEtimeConstantTerm = ps.getParameter<double>("EEtimeConstantTerm");
  auto EBtimeNconst = ps.getParameter<double>("EBtimeNconst");
  auto EEtimeNconst = ps.getParameter<double>("EEtimeNconst");

  auto outOfTimeThreshG12pEB = ps.getParameter<double>("outOfTimeThresholdGain12pEB");
  auto outOfTimeThreshG12mEB = ps.getParameter<double>("outOfTimeThresholdGain12mEB");
  auto outOfTimeThreshG61pEB = ps.getParameter<double>("outOfTimeThresholdGain61pEB");
  auto outOfTimeThreshG61mEB = ps.getParameter<double>("outOfTimeThresholdGain61mEB");
  auto outOfTimeThreshG12pEE = ps.getParameter<double>("outOfTimeThresholdGain12pEE");
  auto outOfTimeThreshG12mEE = ps.getParameter<double>("outOfTimeThresholdGain12mEE");
  auto outOfTimeThreshG61pEE = ps.getParameter<double>("outOfTimeThresholdGain61pEE");
  auto outOfTimeThreshG61mEE = ps.getParameter<double>("outOfTimeThresholdGain61mEE");
  auto amplitudeThreshEB = ps.getParameter<double>("amplitudeThresholdEB");
  auto amplitudeThreshEE = ps.getParameter<double>("amplitudeThresholdEE");

  // max number of digis to allocate for
  maxNumberHits_ = ps.getParameter<uint32_t>("maxNumberHits");

  // switch to run timing computation kernels
  configParameters_.shouldRunTimingComputation = ps.getParameter<bool>("shouldRunTimingComputation");

  // minimize kernel launch conf
  auto threadsMinimize = ps.getParameter<std::vector<uint32_t>>("kernelMinimizeThreads");
  configParameters_.kernelMinimizeThreads[0] = threadsMinimize[0];
  configParameters_.kernelMinimizeThreads[1] = threadsMinimize[1];
  configParameters_.kernelMinimizeThreads[2] = threadsMinimize[2];

  //
  // configuration and physics parameters: done once
  // assume there is a single device
  // use sync copying
  //

  // amplitude fit parameters copying
  cudaCheck(cudaMalloc((void**)&configParameters_.amplitudeFitParametersEB,
                       sizeof(ecal::multifit::ConfigurationParameters::type) * EBamplitudeFitParameters.size()));
  cudaCheck(cudaMemcpy(configParameters_.amplitudeFitParametersEB,
                       EBamplitudeFitParameters.data(),
                       EBamplitudeFitParameters.size() * sizeof(ecal::multifit::ConfigurationParameters::type),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMalloc((void**)&configParameters_.amplitudeFitParametersEE,
                       sizeof(ecal::multifit::ConfigurationParameters::type) * EEamplitudeFitParameters.size()));
  cudaCheck(cudaMemcpy(configParameters_.amplitudeFitParametersEE,
                       EEamplitudeFitParameters.data(),
                       EEamplitudeFitParameters.size() * sizeof(ecal::multifit::ConfigurationParameters::type),
                       cudaMemcpyHostToDevice));

  // time fit parameters and limits
  configParameters_.timeFitParametersSizeEB = EBtimeFitParameters.size();
  configParameters_.timeFitParametersSizeEE = EEtimeFitParameters.size();
  configParameters_.timeFitLimitsFirstEB = EBtimeFitLimits.first;
  configParameters_.timeFitLimitsSecondEB = EBtimeFitLimits.second;
  configParameters_.timeFitLimitsFirstEE = EEtimeFitLimits.first;
  configParameters_.timeFitLimitsSecondEE = EEtimeFitLimits.second;
  cudaCheck(cudaMalloc((void**)&configParameters_.timeFitParametersEB,
                       sizeof(ecal::multifit::ConfigurationParameters::type) * EBtimeFitParameters.size()));
  cudaCheck(cudaMemcpy(configParameters_.timeFitParametersEB,
                       EBtimeFitParameters.data(),
                       EBtimeFitParameters.size() * sizeof(ecal::multifit::ConfigurationParameters::type),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMalloc((void**)&configParameters_.timeFitParametersEE,
                       sizeof(ecal::multifit::ConfigurationParameters::type) * EEtimeFitParameters.size()));
  cudaCheck(cudaMemcpy(configParameters_.timeFitParametersEE,
                       EEtimeFitParameters.data(),
                       EEtimeFitParameters.size() * sizeof(ecal::multifit::ConfigurationParameters::type),
                       cudaMemcpyHostToDevice));

  // time constant terms
  configParameters_.timeConstantTermEB = EBtimeConstantTerm;
  configParameters_.timeConstantTermEE = EEtimeConstantTerm;

  // time N const
  configParameters_.timeNconstEB = EBtimeNconst;
  configParameters_.timeNconstEE = EEtimeNconst;

  // amplitude threshold for time flags
  configParameters_.amplitudeThreshEB = amplitudeThreshEB;
  configParameters_.amplitudeThreshEE = amplitudeThreshEE;

  // out of time thresholds gain-dependent
  configParameters_.outOfTimeThreshG12pEB = outOfTimeThreshG12pEB;
  configParameters_.outOfTimeThreshG12pEE = outOfTimeThreshG12pEE;
  configParameters_.outOfTimeThreshG61pEB = outOfTimeThreshG61pEB;
  configParameters_.outOfTimeThreshG61pEE = outOfTimeThreshG61pEE;
  configParameters_.outOfTimeThreshG12mEB = outOfTimeThreshG12mEB;
  configParameters_.outOfTimeThreshG12mEE = outOfTimeThreshG12mEE;
  configParameters_.outOfTimeThreshG61mEB = outOfTimeThreshG61mEB;
  configParameters_.outOfTimeThreshG61mEE = outOfTimeThreshG61mEE;

  // allocate event output data
  eventOutputDataGPU_.allocate(configParameters_, maxNumberHits_);

  // allocate scratch data for gpu
  eventDataForScratchGPU_.allocate(configParameters_, maxNumberHits_);
}

EcalUncalibRecHitProducerGPU::~EcalUncalibRecHitProducerGPU() {
  //
  // assume single device for now
  //

  if (configParameters_.amplitudeFitParametersEB) {
    // configuration parameters
    cudaCheck(cudaFree(configParameters_.amplitudeFitParametersEB));
    cudaCheck(cudaFree(configParameters_.amplitudeFitParametersEE));
    cudaCheck(cudaFree(configParameters_.timeFitParametersEB));
    cudaCheck(cudaFree(configParameters_.timeFitParametersEE));

    // free event ouput data
    eventOutputDataGPU_.deallocate(configParameters_);

    // free event scratch data
    eventDataForScratchGPU_.deallocate(configParameters_);
  }
}

void EcalUncalibRecHitProducerGPU::acquire(edm::Event const& event,
                                           edm::EventSetup const& setup,
                                           edm::WaitingTaskWithArenaHolder holder) {
  // cuda products
  auto const& ebDigisProduct = event.get(digisTokenEB_);
  auto const& eeDigisProduct = event.get(digisTokenEE_);

  // raii
  cms::cuda::ScopedContextAcquire ctx{ebDigisProduct, std::move(holder), cudaState_};

  // get actual obj
  auto const& ebDigis = ctx.get(ebDigisProduct);
  auto const& eeDigis = ctx.get(eeDigisProduct);
  ecal::multifit::EventInputDataGPU inputDataGPU{ebDigis, eeDigis};
  neb_ = ebDigis.ndigis;
  nee_ = eeDigis.ndigis;

  // conditions
  setup.get<EcalPedestalsRcd>().get(pedestalsHandle_);
  setup.get<EcalGainRatiosRcd>().get(gainRatiosHandle_);
  setup.get<EcalPulseShapesRcd>().get(pulseShapesHandle_);
  setup.get<EcalPulseCovariancesRcd>().get(pulseCovariancesHandle_);
  setup.get<EcalSamplesCorrelationRcd>().get(samplesCorrelationHandle_);
  setup.get<EcalTimeBiasCorrectionsRcd>().get(timeBiasCorrectionsHandle_);
  setup.get<EcalTimeCalibConstantsRcd>().get(timeCalibConstantsHandle_);
  setup.get<EcalSampleMaskRcd>().get(sampleMaskHandle_);
  setup.get<EcalTimeOffsetConstantRcd>().get(timeOffsetConstantHandle_);

  auto const& pedProduct = pedestalsHandle_->getProduct(ctx.stream());
  auto const& gainsProduct = gainRatiosHandle_->getProduct(ctx.stream());
  auto const& pulseShapesProduct = pulseShapesHandle_->getProduct(ctx.stream());
  auto const& pulseCovariancesProduct = pulseCovariancesHandle_->getProduct(ctx.stream());
  auto const& samplesCorrelationProduct = samplesCorrelationHandle_->getProduct(ctx.stream());
  auto const& timeBiasCorrectionsProduct = timeBiasCorrectionsHandle_->getProduct(ctx.stream());
  auto const& timeCalibConstantsProduct = timeCalibConstantsHandle_->getProduct(ctx.stream());

  // bundle up conditions
  ecal::multifit::ConditionsProducts conditions{pedProduct,
                                                gainsProduct,
                                                pulseShapesProduct,
                                                pulseCovariancesProduct,
                                                samplesCorrelationProduct,
                                                timeBiasCorrectionsProduct,
                                                timeCalibConstantsProduct,
                                                *sampleMaskHandle_,
                                                *timeOffsetConstantHandle_,
                                                timeCalibConstantsHandle_->getOffset()};

  //
  // schedule algorithms
  //
  ecal::multifit::entryPoint(
      inputDataGPU, eventOutputDataGPU_, eventDataForScratchGPU_, conditions, configParameters_, ctx.stream());
}

void EcalUncalibRecHitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  //DurationMeasurer<std::chrono::milliseconds> timer{std::string{"produce duration"}};
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  // copy construct output collections
  // note, output collections do not own device memory!
  ecal::UncalibratedRecHit<ecal::Tag::ptr> ebRecHits{eventOutputDataGPU_}, eeRecHits{eventOutputDataGPU_};

  // set the size of eb and ee
  ebRecHits.size = neb_;
  eeRecHits.size = nee_;

  // shift ptrs for ee
  eeRecHits.amplitudesAll += neb_ * EcalDataFrame::MAXSAMPLES;
  eeRecHits.amplitude += neb_;
  eeRecHits.chi2 += neb_;
  eeRecHits.pedestal += neb_;
  eeRecHits.did += neb_;
  eeRecHits.flags += neb_;
  if (configParameters_.shouldRunTimingComputation) {
    eeRecHits.jitter += neb_;
    eeRecHits.jitterError += neb_;
  }

  // put into the event
  ctx.emplace(event, recHitsTokenEB_, std::move(ebRecHits));
  ctx.emplace(event, recHitsTokenEE_, std::move(eeRecHits));
}

DEFINE_FWK_MODULE(EcalUncalibRecHitProducerGPU);

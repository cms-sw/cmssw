// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEvent.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
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

  void transferToHost(RecHitType& ebRecHits, RecHitType& eeRecHits, cudaStream_t cudaStream);

private:
  edm::EDGetTokenT<EBDigiCollection> digisTokenEB_;
  edm::EDGetTokenT<EEDigiCollection> digisTokenEE_;

  std::string recHitsLabelEB_, recHitsLabelEE_;

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
  ecal::multifit::EventInputDataGPU eventInputDataGPU_;
  ecal::multifit::EventOutputDataGPU eventOutputDataGPU_;
  ecal::multifit::EventDataForScratchGPU eventDataForScratchGPU_;
  bool shouldTransferToHost_{true};

  CUDAContextState cudaState_;

  std::unique_ptr<ecal::UncalibratedRecHit<ecal::Tag::soa>> ebRecHits_{nullptr}, eeRecHits_{nullptr};

  uint32_t maxNumberHits_;
};

void EcalUncalibRecHitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digisLabelEB", edm::InputTag("ecalDigis", "ebDigis"));
  desc.add<edm::InputTag>("digisLabelEE", edm::InputTag("ecalDigis", "eeDigis"));

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
  desc.add<bool>("shouldTransferToHost", true);
  desc.add<std::vector<uint32_t>>("kernelMinimizeThreads", {32, 1, 1});
  // ---- default false or true? It was set to true, but at HLT it is false
  desc.add<bool>("shouldRunTimingComputation", false);
  std::string label = "ecalUncalibRecHitProducerGPU";
  confDesc.add(label, desc);
}

EcalUncalibRecHitProducerGPU::EcalUncalibRecHitProducerGPU(const edm::ParameterSet& ps) {
  digisTokenEB_ = consumes<EBDigiCollection>(ps.getParameter<edm::InputTag>("digisLabelEB"));
  digisTokenEE_ = consumes<EEDigiCollection>(ps.getParameter<edm::InputTag>("digisLabelEE"));

  recHitsLabelEB_ = ps.getParameter<std::string>("recHitsLabelEB");
  recHitsLabelEE_ = ps.getParameter<std::string>("recHitsLabelEE");

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

  // transfer to host switch
  shouldTransferToHost_ = ps.getParameter<bool>("shouldTransferToHost");

  // switch to run timing computation kernels
  configParameters_.shouldRunTimingComputation = ps.getParameter<bool>("shouldRunTimingComputation");

  // minimize kernel launch conf
  auto threadsMinimize = ps.getParameter<std::vector<uint32_t>>("kernelMinimizeThreads");
  configParameters_.kernelMinimizeThreads[0] = threadsMinimize[0];
  configParameters_.kernelMinimizeThreads[1] = threadsMinimize[1];
  configParameters_.kernelMinimizeThreads[2] = threadsMinimize[2];

  produces<ecal::SoAUncalibratedRecHitCollection>(recHitsLabelEB_);
  produces<ecal::SoAUncalibratedRecHitCollection>(recHitsLabelEE_);

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

  // allocate event input data
  eventInputDataGPU_.allocate(maxNumberHits_);

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

    // free event input data
    eventInputDataGPU_.deallocate();

    // free event ouput data
    eventOutputDataGPU_.deallocate(configParameters_);

    // free event scratch data
    eventDataForScratchGPU_.deallocate(configParameters_);
  }
}

void EcalUncalibRecHitProducerGPU::acquire(edm::Event const& event,
                                           edm::EventSetup const& setup,
                                           edm::WaitingTaskWithArenaHolder holder) {
  //DurationMeasurer<std::chrono::milliseconds> timer{std::string{"acquire duration"}};

  // raii
  CUDAScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

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
  // retrieve collections
  //
  edm::Handle<EBDigiCollection> ebDigis;
  edm::Handle<EEDigiCollection> eeDigis;
  event.getByToken(digisTokenEB_, ebDigis);
  event.getByToken(digisTokenEE_, eeDigis);

  ecal::multifit::EventInputDataCPU eventInputDataCPU{*ebDigis, *eeDigis};

  //
  // schedule algorithms
  //
  ecal::multifit::entryPoint(eventInputDataCPU,
                             eventInputDataGPU_,
                             eventOutputDataGPU_,
                             eventDataForScratchGPU_,
                             conditions,
                             configParameters_,
                             ctx.stream());

  ebRecHits_ = std::make_unique<ecal::UncalibratedRecHit<ecal::Tag::soa>>();
  eeRecHits_ = std::make_unique<ecal::UncalibratedRecHit<ecal::Tag::soa>>();

  if (shouldTransferToHost_) {
    // allocate for the result while kernels are running
    ebRecHits_->resize(ebDigis->size());
    eeRecHits_->resize(eeDigis->size());

    // det ids are host copy only - no need to run device -> host
    std::memcpy(ebRecHits_->did.data(), ebDigis->ids().data(), ebDigis->ids().size() * sizeof(uint32_t));
    std::memcpy(eeRecHits_->did.data(), eeDigis->ids().data(), eeDigis->ids().size() * sizeof(uint32_t));
  }
}

void EcalUncalibRecHitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  //DurationMeasurer<std::chrono::milliseconds> timer{std::string{"produce duration"}};
  CUDAScopedContextProduce ctx{cudaState_};

  if (shouldTransferToHost_) {
    // rec hits objects were not originally member variables
    transferToHost(*ebRecHits_, *eeRecHits_, ctx.stream());

    // TODO
    // for now just sync on the host when transferring back products
    cudaStreamSynchronize(ctx.stream());
  }

  event.put(std::move(ebRecHits_), recHitsLabelEB_);
  event.put(std::move(eeRecHits_), recHitsLabelEE_);
}

void EcalUncalibRecHitProducerGPU::transferToHost(RecHitType& ebRecHits,
                                                  RecHitType& eeRecHits,
                                                  cudaStream_t cudaStream) {
  cudaCheck(cudaMemcpyAsync(ebRecHits.amplitude.data(),
                            eventOutputDataGPU_.amplitude,
                            ebRecHits.amplitude.size() * sizeof(ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(eeRecHits.amplitude.data(),
                            eventOutputDataGPU_.amplitude + ebRecHits.amplitude.size(),
                            eeRecHits.amplitude.size() * sizeof(ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            cudaStream));

  cudaCheck(cudaMemcpyAsync(ebRecHits.pedestal.data(),
                            eventOutputDataGPU_.pedestal,
                            ebRecHits.pedestal.size() * sizeof(ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(eeRecHits.pedestal.data(),
                            eventOutputDataGPU_.pedestal + ebRecHits.pedestal.size(),
                            eeRecHits.pedestal.size() * sizeof(ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            cudaStream));

  cudaCheck(cudaMemcpyAsync(ebRecHits.chi2.data(),
                            eventOutputDataGPU_.chi2,
                            ebRecHits.chi2.size() * sizeof(ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(eeRecHits.chi2.data(),
                            eventOutputDataGPU_.chi2 + ebRecHits.chi2.size(),
                            eeRecHits.chi2.size() * sizeof(ecal::reco::StorageScalarType),
                            cudaMemcpyDeviceToHost,
                            cudaStream));

  if (configParameters_.shouldRunTimingComputation) {
    cudaCheck(cudaMemcpyAsync(ebRecHits.jitter.data(),
                              eventOutputDataGPU_.jitter,
                              ebRecHits.jitter.size() * sizeof(ecal::reco::StorageScalarType),
                              cudaMemcpyDeviceToHost,
                              cudaStream));
    cudaCheck(cudaMemcpyAsync(eeRecHits.jitter.data(),
                              eventOutputDataGPU_.jitter + ebRecHits.jitter.size(),
                              eeRecHits.jitter.size() * sizeof(ecal::reco::StorageScalarType),
                              cudaMemcpyDeviceToHost,
                              cudaStream));

    cudaCheck(cudaMemcpyAsync(ebRecHits.jitterError.data(),
                              eventOutputDataGPU_.jitterError,
                              ebRecHits.jitterError.size() * sizeof(ecal::reco::StorageScalarType),
                              cudaMemcpyDeviceToHost,
                              cudaStream));
    cudaCheck(cudaMemcpyAsync(eeRecHits.jitterError.data(),
                              eventOutputDataGPU_.jitterError + ebRecHits.jitterError.size(),
                              eeRecHits.jitterError.size() * sizeof(ecal::reco::StorageScalarType),
                              cudaMemcpyDeviceToHost,
                              cudaStream));
  }

  cudaCheck(cudaMemcpyAsync(ebRecHits.flags.data(),
                            eventOutputDataGPU_.flags,
                            ebRecHits.flags.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(eeRecHits.flags.data(),
                            eventOutputDataGPU_.flags + ebRecHits.flags.size(),
                            eeRecHits.flags.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost,
                            cudaStream));

  cudaCheck(cudaMemcpyAsync(ebRecHits.amplitudesAll.data(),
                            eventOutputDataGPU_.amplitudesAll,
                            ebRecHits.amplitudesAll.size() * sizeof(ecal::reco::ComputationScalarType),
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(eeRecHits.amplitudesAll.data(),
                            eventOutputDataGPU_.amplitudesAll + ebRecHits.amplitudesAll.size(),
                            eeRecHits.amplitudesAll.size() * sizeof(ecal::reco::ComputationScalarType),
                            cudaMemcpyDeviceToHost,
                            cudaStream));
}

DEFINE_FWK_MODULE(EcalUncalibRecHitProducerGPU);

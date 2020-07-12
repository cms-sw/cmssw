#include <iostream>

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalGainRatiosGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPedestalsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseCovariancesGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseShapesGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSamplesCorrelationGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalTimeBiasCorrectionsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalTimeCalibConstantsGPU.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalMultifitParametersGPU.h"

#include "Common.h"
#include "DeclsForKernels.h"
#include "EcalUncalibRecHitMultiFitAlgo_gpu_new.h"
#include "EcalMultifitParametersGPURecord.h"

class EcalUncalibRecHitProducerGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalUncalibRecHitProducerGPU(edm::ParameterSet const& ps);
  ~EcalUncalibRecHitProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  using InputProduct = cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<InputProduct> digisTokenEB_, digisTokenEE_;
  using OutputProduct = cms::cuda::Product<ecal::UncalibratedRecHit<calo::common::DevStoragePolicy>>;
  edm::EDPutTokenT<OutputProduct> recHitsTokenEB_, recHitsTokenEE_;

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
  edm::ESHandle<EcalMultifitParametersGPU> multifitParametersHandle_;

  // configuration parameters
  ecal::multifit::ConfigurationParameters configParameters_;

  // event data
  ecal::multifit::EventOutputDataGPU eventOutputDataGPU_;

  cms::cuda::ContextState cudaState_;

  uint32_t neb_, nee_;
};

void EcalUncalibRecHitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digisLabelEB", edm::InputTag("ecalRawToDigiGPU", "ebDigisGPU"));
  desc.add<edm::InputTag>("digisLabelEE", edm::InputTag("ecalRawToDigiGPU", "eeDigisGPU"));

  desc.add<std::string>("recHitsLabelEB", "EcalUncalibRecHitsEB");
  desc.add<std::string>("recHitsLabelEE", "EcalUncalibRecHitsEE");

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
    : digisTokenEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("digisLabelEB"))},
      digisTokenEE_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("digisLabelEE"))},
      recHitsTokenEB_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsLabelEB"))},
      recHitsTokenEE_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsLabelEE"))} {
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
  configParameters_.maxNumberHits = ps.getParameter<uint32_t>("maxNumberHits");

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

  // time fit parameters and limits
  configParameters_.timeFitLimitsFirstEB = EBtimeFitLimits.first;
  configParameters_.timeFitLimitsSecondEB = EBtimeFitLimits.second;
  configParameters_.timeFitLimitsFirstEE = EEtimeFitLimits.first;
  configParameters_.timeFitLimitsSecondEE = EEtimeFitLimits.second;

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
}

EcalUncalibRecHitProducerGPU::~EcalUncalibRecHitProducerGPU() {}

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
  neb_ = ebDigis.size;
  nee_ = eeDigis.size;

  if ((neb_ + nee_) > configParameters_.maxNumberHits) {
    edm::LogError("EcalUncalibRecHitProducerGPU") << "max number of channels exceeded. See options 'maxNumberHits' ";
  }

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
  setup.get<EcalMultifitParametersGPURecord>().get(multifitParametersHandle_);

  auto const& pedProduct = pedestalsHandle_->getProduct(ctx.stream());
  auto const& gainsProduct = gainRatiosHandle_->getProduct(ctx.stream());
  auto const& pulseShapesProduct = pulseShapesHandle_->getProduct(ctx.stream());
  auto const& pulseCovariancesProduct = pulseCovariancesHandle_->getProduct(ctx.stream());
  auto const& samplesCorrelationProduct = samplesCorrelationHandle_->getProduct(ctx.stream());
  auto const& timeBiasCorrectionsProduct = timeBiasCorrectionsHandle_->getProduct(ctx.stream());
  auto const& timeCalibConstantsProduct = timeCalibConstantsHandle_->getProduct(ctx.stream());
  auto const& multifitParametersProduct = multifitParametersHandle_->getProduct(ctx.stream());

  // assign ptrs/values: this is done not to change how things look downstream
  configParameters_.amplitudeFitParametersEB = multifitParametersProduct.amplitudeFitParametersEB;
  configParameters_.amplitudeFitParametersEE = multifitParametersProduct.amplitudeFitParametersEE;
  configParameters_.timeFitParametersEB = multifitParametersProduct.timeFitParametersEB;
  configParameters_.timeFitParametersEE = multifitParametersProduct.timeFitParametersEE;
  configParameters_.timeFitParametersSizeEB = multifitParametersHandle_->getValues()[2].get().size();
  configParameters_.timeFitParametersSizeEE = multifitParametersHandle_->getValues()[3].get().size();

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
                                                timeCalibConstantsHandle_->getOffset(),
                                                multifitParametersProduct};

  // dev mem
  eventOutputDataGPU_.allocate(configParameters_, ctx.stream());

  // scratch mem
  ecal::multifit::EventDataForScratchGPU eventDataForScratchGPU;
  eventDataForScratchGPU.allocate(configParameters_, ctx.stream());

  //
  // schedule algorithms
  //
  ecal::multifit::entryPoint(
      inputDataGPU, eventOutputDataGPU_, eventDataForScratchGPU, conditions, configParameters_, ctx.stream());
}

void EcalUncalibRecHitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  //DurationMeasurer<std::chrono::milliseconds> timer{std::string{"produce duration"}};
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  // set the size of eb and ee
  eventOutputDataGPU_.recHitsEB.size = neb_;
  eventOutputDataGPU_.recHitsEE.size = nee_;

  // put into the event
  ctx.emplace(event, recHitsTokenEB_, std::move(eventOutputDataGPU_.recHitsEB));
  ctx.emplace(event, recHitsTokenEE_, std::move(eventOutputDataGPU_.recHitsEE));
}

DEFINE_FWK_MODULE(EcalUncalibRecHitProducerGPU);

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
#include "CondFormats/EcalObjects/interface/EcalGainRatiosGPU.h"
#include "CondFormats/EcalObjects/interface/EcalMultifitParametersGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPedestalsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariancesGPU.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapesGPU.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelationGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrectionsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstantsGPU.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "DeclsForKernels.h"
#include "EcalUncalibRecHitMultiFitAlgoGPU.h"

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
  const edm::EDGetTokenT<InputProduct> digisTokenEB_, digisTokenEE_;
  using OutputProduct = cms::cuda::Product<ecal::UncalibratedRecHit<calo::common::DevStoragePolicy>>;
  const edm::EDPutTokenT<OutputProduct> recHitsTokenEB_, recHitsTokenEE_;

  // conditions tokens
  const edm::ESGetToken<EcalPedestalsGPU, EcalPedestalsRcd> pedestalsToken_;
  const edm::ESGetToken<EcalGainRatiosGPU, EcalGainRatiosRcd> gainRatiosToken_;
  const edm::ESGetToken<EcalPulseShapesGPU, EcalPulseShapesRcd> pulseShapesToken_;
  const edm::ESGetToken<EcalPulseCovariancesGPU, EcalPulseCovariancesRcd> pulseCovariancesToken_;
  const edm::ESGetToken<EcalSamplesCorrelationGPU, EcalSamplesCorrelationRcd> samplesCorrelationToken_;
  const edm::ESGetToken<EcalTimeBiasCorrectionsGPU, EcalTimeBiasCorrectionsRcd> timeBiasCorrectionsToken_;
  const edm::ESGetToken<EcalTimeCalibConstantsGPU, EcalTimeCalibConstantsRcd> timeCalibConstantsToken_;
  const edm::ESGetToken<EcalSampleMask, EcalSampleMaskRcd> sampleMaskToken_;
  const edm::ESGetToken<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd> timeOffsetConstantToken_;
  const edm::ESGetToken<EcalMultifitParametersGPU, JobConfigurationGPURecord> multifitParametersToken_;

  // configuration parameters
  ecal::multifit::ConfigurationParameters configParameters_;

  // event data
  ecal::multifit::EventOutputDataGPU eventOutputDataGPU_;

  cms::cuda::ContextState cudaState_;

  uint32_t neb_, nee_;
};

void EcalUncalibRecHitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digisLabelEB", edm::InputTag("ecalRawToDigiGPU", "ebDigis"));
  desc.add<edm::InputTag>("digisLabelEE", edm::InputTag("ecalRawToDigiGPU", "eeDigis"));

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
  desc.add<uint32_t>("maxNumberHitsEB", 61200);
  desc.add<uint32_t>("maxNumberHitsEE", 14648);
  desc.addUntracked<std::vector<uint32_t>>("kernelMinimizeThreads", {32, 1, 1});
  desc.add<bool>("shouldRunTimingComputation", true);
  confDesc.addWithDefaultLabel(desc);
}

EcalUncalibRecHitProducerGPU::EcalUncalibRecHitProducerGPU(const edm::ParameterSet& ps)
    : digisTokenEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("digisLabelEB"))},
      digisTokenEE_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("digisLabelEE"))},
      recHitsTokenEB_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsLabelEB"))},
      recHitsTokenEE_{produces<OutputProduct>(ps.getParameter<std::string>("recHitsLabelEE"))},
      pedestalsToken_{esConsumes<EcalPedestalsGPU, EcalPedestalsRcd>()},
      gainRatiosToken_{esConsumes<EcalGainRatiosGPU, EcalGainRatiosRcd>()},
      pulseShapesToken_{esConsumes<EcalPulseShapesGPU, EcalPulseShapesRcd>()},
      pulseCovariancesToken_{esConsumes<EcalPulseCovariancesGPU, EcalPulseCovariancesRcd>()},
      samplesCorrelationToken_{esConsumes<EcalSamplesCorrelationGPU, EcalSamplesCorrelationRcd>()},
      timeBiasCorrectionsToken_{esConsumes<EcalTimeBiasCorrectionsGPU, EcalTimeBiasCorrectionsRcd>()},
      timeCalibConstantsToken_{esConsumes<EcalTimeCalibConstantsGPU, EcalTimeCalibConstantsRcd>()},
      sampleMaskToken_{esConsumes<EcalSampleMask, EcalSampleMaskRcd>()},
      timeOffsetConstantToken_{esConsumes<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd>()},
      multifitParametersToken_{esConsumes<EcalMultifitParametersGPU, JobConfigurationGPURecord>()} {
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
  configParameters_.maxNumberHitsEB = ps.getParameter<uint32_t>("maxNumberHitsEB");
  configParameters_.maxNumberHitsEE = ps.getParameter<uint32_t>("maxNumberHitsEE");

  // switch to run timing computation kernels
  configParameters_.shouldRunTimingComputation = ps.getParameter<bool>("shouldRunTimingComputation");

  // minimize kernel launch conf
  auto threadsMinimize = ps.getUntrackedParameter<std::vector<uint32_t>>("kernelMinimizeThreads");
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

  // stop here if there are no digis
  if (neb_ + nee_ == 0)
    return;

  if ((neb_ > configParameters_.maxNumberHitsEB) || (nee_ > configParameters_.maxNumberHitsEE)) {
    edm::LogError("EcalUncalibRecHitProducerGPU")
        << "max number of channels exceeded. See options 'maxNumberHitsEB and maxNumberHitsEE' ";
  }

  // conditions
  auto const& timeCalibConstantsData = setup.getData(timeCalibConstantsToken_);
  auto const& sampleMaskData = setup.getData(sampleMaskToken_);
  auto const& timeOffsetConstantData = setup.getData(timeOffsetConstantToken_);
  auto const& multifitParametersData = setup.getData(multifitParametersToken_);

  auto const& pedestals = setup.getData(pedestalsToken_).getProduct(ctx.stream());
  auto const& gainRatios = setup.getData(gainRatiosToken_).getProduct(ctx.stream());
  auto const& pulseShapes = setup.getData(pulseShapesToken_).getProduct(ctx.stream());
  auto const& pulseCovariances = setup.getData(pulseCovariancesToken_).getProduct(ctx.stream());
  auto const& samplesCorrelation = setup.getData(samplesCorrelationToken_).getProduct(ctx.stream());
  auto const& timeBiasCorrections = setup.getData(timeBiasCorrectionsToken_).getProduct(ctx.stream());
  auto const& timeCalibConstants = timeCalibConstantsData.getProduct(ctx.stream());
  auto const& multifitParameters = multifitParametersData.getProduct(ctx.stream());

  // assign ptrs/values: this is done not to change how things look downstream
  configParameters_.amplitudeFitParametersEB = multifitParameters.amplitudeFitParametersEB.get();
  configParameters_.amplitudeFitParametersEE = multifitParameters.amplitudeFitParametersEE.get();
  configParameters_.timeFitParametersEB = multifitParameters.timeFitParametersEB.get();
  configParameters_.timeFitParametersEE = multifitParameters.timeFitParametersEE.get();
  configParameters_.timeFitParametersSizeEB = multifitParametersData.getValues()[2].get().size();
  configParameters_.timeFitParametersSizeEE = multifitParametersData.getValues()[3].get().size();

  // bundle up conditions
  ecal::multifit::ConditionsProducts conditions{pedestals,
                                                gainRatios,
                                                pulseShapes,
                                                pulseCovariances,
                                                samplesCorrelation,
                                                timeBiasCorrections,
                                                timeCalibConstants,
                                                sampleMaskData,
                                                timeOffsetConstantData,
                                                timeCalibConstantsData.getOffset(),
                                                multifitParameters};

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

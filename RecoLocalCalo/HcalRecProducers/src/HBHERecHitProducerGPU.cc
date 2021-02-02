#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "SimpleAlgoGPU.h"

class HBHERecHitProducerGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit HBHERecHitProducerGPU(edm::ParameterSet const&);
  ~HBHERecHitProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

  using IProductTypef01 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor1, calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<IProductTypef01> digisTokenF01HE_;

  using IProductTypef5 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor5, calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<IProductTypef5> digisTokenF5HB_;

  using IProductTypef3 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor3, calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<IProductTypef3> digisTokenF3HB_;

  using RecHitType = hcal::RecHitCollection<calo::common::DevStoragePolicy>;
  using OProductType = cms::cuda::Product<RecHitType>;
  edm::EDPutTokenT<OProductType> rechitsM0Token_;

  hcal::reconstruction::ConfigParameters configParameters_;
  hcal::reconstruction::OutputDataGPU outputGPU_;
  cms::cuda::ContextState cudaState_;
};

HBHERecHitProducerGPU::HBHERecHitProducerGPU(edm::ParameterSet const& ps)
    : digisTokenF01HE_{consumes<IProductTypef01>(ps.getParameter<edm::InputTag>("digisLabelF01HE"))},
      digisTokenF5HB_{consumes<IProductTypef5>(ps.getParameter<edm::InputTag>("digisLabelF5HB"))},
      digisTokenF3HB_{consumes<IProductTypef3>(ps.getParameter<edm::InputTag>("digisLabelF3HB"))},
      rechitsM0Token_{produces<OProductType>(ps.getParameter<std::string>("recHitsLabelM0HBHE"))} {
  configParameters_.maxChannels = ps.getParameter<uint32_t>("maxChannels");
  configParameters_.maxTimeSamples = ps.getParameter<uint32_t>("maxTimeSamples");
  configParameters_.kprep1dChannelsPerBlock = ps.getParameter<uint32_t>("kprep1dChannelsPerBlock");
  configParameters_.sipmQTSShift = ps.getParameter<int>("sipmQTSShift");
  configParameters_.sipmQNTStoSum = ps.getParameter<int>("sipmQNTStoSum");
  configParameters_.firstSampleShift = ps.getParameter<int>("firstSampleShift");
  configParameters_.useEffectivePedestals = ps.getParameter<bool>("useEffectivePedestals");

  configParameters_.meanTime = ps.getParameter<double>("meanTime");
  configParameters_.timeSigmaSiPM = ps.getParameter<double>("timeSigmaSiPM");
  configParameters_.timeSigmaHPD = ps.getParameter<double>("timeSigmaHPD");
  configParameters_.ts4Thresh = ps.getParameter<double>("ts4Thresh");

  configParameters_.applyTimeSlew = ps.getParameter<bool>("applyTimeSlew");
  auto const tzeroValues = ps.getParameter<std::vector<double>>("tzeroTimeSlewParameters");
  auto const slopeValues = ps.getParameter<std::vector<double>>("slopeTimeSlewParameters");
  auto const tmaxValues = ps.getParameter<std::vector<double>>("tmaxTimeSlewParameters");

  configParameters_.tzeroTimeSlew = tzeroValues[HcalTimeSlew::Medium];
  configParameters_.slopeTimeSlew = slopeValues[HcalTimeSlew::Medium];
  configParameters_.tmaxTimeSlew = tmaxValues[HcalTimeSlew::Medium];

  auto threadsMinimize = ps.getParameter<std::vector<uint32_t>>("kernelMinimizeThreads");
  configParameters_.kernelMinimizeThreads[0] = threadsMinimize[0];
  configParameters_.kernelMinimizeThreads[1] = threadsMinimize[1];
  configParameters_.kernelMinimizeThreads[2] = threadsMinimize[2];
}

HBHERecHitProducerGPU::~HBHERecHitProducerGPU() {}

void HBHERecHitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
  edm::ParameterSetDescription desc;
  desc.add<uint32_t>("maxChannels", 10000u);
  desc.add<uint32_t>("maxTimeSamples", 10);
  desc.add<uint32_t>("kprep1dChannelsPerBlock", 32);
  desc.add<edm::InputTag>("digisLabelF01HE", edm::InputTag{"hcalRawToDigiGPU", "f01HEDigisGPU"});
  desc.add<edm::InputTag>("digisLabelF5HB", edm::InputTag{"hcalRawToDigiGPU", "f5HBDigisGPU"});
  desc.add<edm::InputTag>("digisLabelF3HB", edm::InputTag{"hcalRawToDigiGPU", "f3HBDigisGPU"});
  desc.add<std::string>("recHitsLabelM0HBHE", "recHitsM0HBHE");
  desc.add<int>("sipmQTSShift", 0);
  desc.add<int>("sipmQNTStoSum", 3);
  desc.add<int>("firstSampleShift", 0);
  desc.add<bool>("useEffectivePedestals", true);

  desc.add<double>("meanTime", 0.f);
  desc.add<double>("timeSigmaSiPM", 2.5f);
  desc.add<double>("timeSigmaHPD", 5.0f);
  desc.add<double>("ts4Thresh", 0.0);

  desc.add<bool>("applyTimeSlew", true);
  desc.add<std::vector<double>>("tzeroTimeSlewParameters", {23.960177, 11.977461, 9.109694});
  desc.add<std::vector<double>>("slopeTimeSlewParameters", {-3.178648, -1.5610227, -1.075824});
  desc.add<std::vector<double>>("tmaxTimeSlewParameters", {16.00, 10.00, 6.25});
  desc.add<std::vector<uint32_t>>("kernelMinimizeThreads", {16, 1, 1});

  cdesc.addWithDefaultLabel(desc);
}

void HBHERecHitProducerGPU::acquire(edm::Event const& event,
                                    edm::EventSetup const& setup,
                                    edm::WaitingTaskWithArenaHolder holder) {
#ifdef HCAL_MAHI_CPUDEBUG
  auto start = std::chrono::high_resolution_clock::now();
#endif

  // input + raii
  auto const& f01HEProduct = event.get(digisTokenF01HE_);
  auto const& f5HBProduct = event.get(digisTokenF5HB_);
  auto const& f3HBProduct = event.get(digisTokenF3HB_);
  cms::cuda::ScopedContextAcquire ctx{f01HEProduct, std::move(holder), cudaState_};
  auto const& f01HEDigis = ctx.get(f01HEProduct);
  auto const& f5HBDigis = ctx.get(f5HBProduct);
  auto const& f3HBDigis = ctx.get(f3HBProduct);

  hcal::reconstruction::InputDataGPU inputGPU{f01HEDigis, f5HBDigis, f3HBDigis};

  // conditions
  edm::ESHandle<HcalRecoParamsWithPulseShapesGPU> recoParamsHandle;
  setup.get<HcalRecoParamsRcd>().get(recoParamsHandle);
  auto const& recoParamsProduct = recoParamsHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalGainWidthsGPU> gainWidthsHandle;
  setup.get<HcalGainWidthsRcd>().get(gainWidthsHandle);
  auto const& gainWidthsProduct = gainWidthsHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalGainsGPU> gainsHandle;
  setup.get<HcalGainsRcd>().get(gainsHandle);
  auto const& gainsProduct = gainsHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalLUTCorrsGPU> lutCorrsHandle;
  setup.get<HcalLUTCorrsRcd>().get(lutCorrsHandle);
  auto const& lutCorrsProduct = lutCorrsHandle->getProduct(ctx.stream());

  // use only 1 depending on useEffectivePedestals
  edm::ESHandle<HcalConvertedPedestalWidthsGPU> pedestalWidthsHandle;
  edm::ESHandle<HcalConvertedEffectivePedestalWidthsGPU> effectivePedestalWidthsHandle;
  setup.get<HcalConvertedPedestalWidthsRcd>().get(effectivePedestalWidthsHandle);
  setup.get<HcalConvertedPedestalWidthsRcd>().get(pedestalWidthsHandle);
  auto const& pedestalWidthsProduct = pedestalWidthsHandle->getProduct(ctx.stream());
  auto const& effectivePedestalWidthsProduct = effectivePedestalWidthsHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalConvertedPedestalsGPU> pedestalsHandle;
  setup.get<HcalConvertedPedestalsRcd>().get(pedestalsHandle);
  auto const& pedestalsProduct = pedestalsHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalConvertedEffectivePedestalsGPU> effectivePedestalsHandle;
  if (configParameters_.useEffectivePedestals)
    setup.get<HcalConvertedPedestalsRcd>().get(effectivePedestalsHandle);
  auto const* effectivePedestalsProduct =
      configParameters_.useEffectivePedestals ? &effectivePedestalsHandle->getProduct(ctx.stream()) : nullptr;

  edm::ESHandle<HcalQIECodersGPU> qieCodersHandle;
  setup.get<HcalQIEDataRcd>().get(qieCodersHandle);
  auto const& qieCodersProduct = qieCodersHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalRespCorrsGPU> respCorrsHandle;
  setup.get<HcalRespCorrsRcd>().get(respCorrsHandle);
  auto const& respCorrsProduct = respCorrsHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalTimeCorrsGPU> timeCorrsHandle;
  setup.get<HcalTimeCorrsRcd>().get(timeCorrsHandle);
  auto const& timeCorrsProduct = timeCorrsHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalQIETypesGPU> qieTypesHandle;
  setup.get<HcalQIETypesRcd>().get(qieTypesHandle);
  auto const& qieTypesProduct = qieTypesHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalTopology> topologyHandle;
  setup.get<HcalRecNumberingRecord>().get(topologyHandle);
  edm::ESHandle<HcalDDDRecConstants> recConstantsHandle;
  setup.get<HcalRecNumberingRecord>().get(recConstantsHandle);

  edm::ESHandle<HcalSiPMParametersGPU> sipmParametersHandle;
  setup.get<HcalSiPMParametersRcd>().get(sipmParametersHandle);
  auto const& sipmParametersProduct = sipmParametersHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalSiPMCharacteristicsGPU> sipmCharacteristicsHandle;
  setup.get<HcalSiPMCharacteristicsRcd>().get(sipmCharacteristicsHandle);
  auto const& sipmCharacteristicsProduct = sipmCharacteristicsHandle->getProduct(ctx.stream());

  edm::ESHandle<HcalMahiPulseOffsetsGPU> pulseOffsetsHandle;
  setup.get<JobConfigurationGPURecord>().get(pulseOffsetsHandle);
  auto const& pulseOffsetsProduct = pulseOffsetsHandle->getProduct(ctx.stream());

  // bundle up conditions
  hcal::reconstruction::ConditionsProducts conditions{gainWidthsProduct,
                                                      gainsProduct,
                                                      lutCorrsProduct,
                                                      pedestalWidthsProduct,
                                                      effectivePedestalWidthsProduct,
                                                      pedestalsProduct,
                                                      qieCodersProduct,
                                                      recoParamsProduct,
                                                      respCorrsProduct,
                                                      timeCorrsProduct,
                                                      qieTypesProduct,
                                                      sipmParametersProduct,
                                                      sipmCharacteristicsProduct,
                                                      effectivePedestalsProduct,
                                                      topologyHandle.product(),
                                                      recConstantsHandle.product(),
                                                      pedestalsHandle->offsetForHashes(),
                                                      pulseOffsetsProduct,
                                                      pulseOffsetsHandle->getValues()};

  // scratch mem on device
  hcal::reconstruction::ScratchDataGPU scratchGPU = {
      cms::cuda::make_device_unique<float[]>(configParameters_.maxChannels * configParameters_.maxTimeSamples,
                                             ctx.stream()),
      cms::cuda::make_device_unique<float[]>(configParameters_.maxChannels * configParameters_.maxTimeSamples,
                                             ctx.stream()),
      cms::cuda::make_device_unique<float[]>(
          configParameters_.maxChannels * configParameters_.maxTimeSamples * configParameters_.maxTimeSamples,
          ctx.stream()),
      cms::cuda::make_device_unique<float[]>(
          configParameters_.maxChannels * configParameters_.maxTimeSamples * configParameters_.maxTimeSamples,
          ctx.stream()),
      cms::cuda::make_device_unique<float[]>(
          configParameters_.maxChannels * configParameters_.maxTimeSamples * configParameters_.maxTimeSamples,
          ctx.stream()),
      cms::cuda::make_device_unique<int8_t[]>(configParameters_.maxChannels, ctx.stream()),
  };

  // output dev mem
  outputGPU_.allocate(configParameters_, ctx.stream());

  hcal::reconstruction::entryPoint(inputGPU, outputGPU_, conditions, scratchGPU, configParameters_, ctx.stream());

#ifdef HCAL_MAHI_CPUDEBUG
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "acquire  duration = " << duration << std::endl;
#endif
}

void HBHERecHitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};
  ctx.emplace(event, rechitsM0Token_, std::move(outputGPU_.recHits));
}

DEFINE_FWK_MODULE(HBHERecHitProducerGPU);

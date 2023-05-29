#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"

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

  const edm::ESGetToken<HcalRecoParamsWithPulseShapesGPU, HcalRecoParamsRcd> recoParamsToken_;
  const edm::ESGetToken<HcalGainWidthsGPU, HcalGainWidthsRcd> gainWidthsToken_;
  const edm::ESGetToken<HcalGainsGPU, HcalGainsRcd> gainsToken_;
  const edm::ESGetToken<HcalLUTCorrsGPU, HcalLUTCorrsRcd> lutCorrsToken_;
  const edm::ESGetToken<HcalConvertedPedestalWidthsGPU, HcalConvertedPedestalWidthsRcd> pedestalWidthsToken_;
  const edm::ESGetToken<HcalConvertedEffectivePedestalWidthsGPU, HcalConvertedPedestalWidthsRcd>
      effectivePedestalWidthsToken_;
  const edm::ESGetToken<HcalConvertedPedestalsGPU, HcalConvertedPedestalsRcd> pedestalsToken_;
  edm::ESGetToken<HcalConvertedEffectivePedestalsGPU, HcalConvertedPedestalsRcd> effectivePedestalsToken_;
  const edm::ESGetToken<HcalQIECodersGPU, HcalQIEDataRcd> qieCodersToken_;
  const edm::ESGetToken<HcalRespCorrsGPU, HcalRespCorrsRcd> respCorrsToken_;
  const edm::ESGetToken<HcalTimeCorrsGPU, HcalTimeCorrsRcd> timeCorrsToken_;
  const edm::ESGetToken<HcalQIETypesGPU, HcalQIETypesRcd> qieTypesToken_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topologyToken_;
  const edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> recConstantsToken_;
  const edm::ESGetToken<HcalSiPMParametersGPU, HcalSiPMParametersRcd> sipmParametersToken_;
  const edm::ESGetToken<HcalSiPMCharacteristicsGPU, HcalSiPMCharacteristicsRcd> sipmCharacteristicsToken_;
  const edm::ESGetToken<HcalChannelQualityGPU, HcalChannelQualityRcd> chQualProductToken_;
  const edm::ESGetToken<HcalMahiPulseOffsetsGPU, JobConfigurationGPURecord> pulseOffsetsToken_;

  hcal::reconstruction::ConfigParameters configParameters_;
  hcal::reconstruction::OutputDataGPU outputGPU_;
  cms::cuda::ContextState cudaState_;
};

HBHERecHitProducerGPU::HBHERecHitProducerGPU(edm::ParameterSet const& ps)
    : digisTokenF01HE_{consumes<IProductTypef01>(ps.getParameter<edm::InputTag>("digisLabelF01HE"))},
      digisTokenF5HB_{consumes<IProductTypef5>(ps.getParameter<edm::InputTag>("digisLabelF5HB"))},
      digisTokenF3HB_{consumes<IProductTypef3>(ps.getParameter<edm::InputTag>("digisLabelF3HB"))},
      rechitsM0Token_{produces<OProductType>(ps.getParameter<std::string>("recHitsLabelM0HBHE"))},
      recoParamsToken_{esConsumes()},
      gainWidthsToken_{esConsumes()},
      gainsToken_{esConsumes()},
      lutCorrsToken_{esConsumes()},
      pedestalWidthsToken_{esConsumes()},
      effectivePedestalWidthsToken_{esConsumes()},
      pedestalsToken_{esConsumes()},
      qieCodersToken_{esConsumes()},
      respCorrsToken_{esConsumes()},
      timeCorrsToken_{esConsumes()},
      qieTypesToken_{esConsumes()},
      topologyToken_{esConsumes()},
      recConstantsToken_{esConsumes()},
      sipmParametersToken_{esConsumes()},
      sipmCharacteristicsToken_{esConsumes()},
      chQualProductToken_{esConsumes()},
      pulseOffsetsToken_{esConsumes()} {
  configParameters_.maxTimeSamples = ps.getParameter<uint32_t>("maxTimeSamples");
  configParameters_.kprep1dChannelsPerBlock = ps.getParameter<uint32_t>("kprep1dChannelsPerBlock");
  configParameters_.sipmQTSShift = ps.getParameter<int>("sipmQTSShift");
  configParameters_.sipmQNTStoSum = ps.getParameter<int>("sipmQNTStoSum");
  configParameters_.firstSampleShift = ps.getParameter<int>("firstSampleShift");
  configParameters_.useEffectivePedestals = ps.getParameter<bool>("useEffectivePedestals");
  if (configParameters_.useEffectivePedestals) {
    effectivePedestalsToken_ = esConsumes();
  }

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
  auto const totalChannels = f01HEDigis.size + f5HBDigis.size + f3HBDigis.size;

  hcal::reconstruction::InputDataGPU inputGPU{f01HEDigis, f5HBDigis, f3HBDigis};

  // conditions
  auto const& recoParamsProduct = setup.getData(recoParamsToken_).getProduct(ctx.stream());

  auto const& gainWidthsProduct = setup.getData(gainWidthsToken_).getProduct(ctx.stream());

  auto const& gainsProduct = setup.getData(gainsToken_).getProduct(ctx.stream());

  auto const& lutCorrsProduct = setup.getData(lutCorrsToken_).getProduct(ctx.stream());

  // use only 1 depending on useEffectivePedestals
  auto const& pedestalWidthsProduct = setup.getData(pedestalWidthsToken_).getProduct(ctx.stream());
  auto const& effectivePedestalWidthsProduct = setup.getData(effectivePedestalWidthsToken_).getProduct(ctx.stream());

  auto const& pedestals = setup.getData(pedestalsToken_);
  auto const& pedestalsProduct = pedestals.getProduct(ctx.stream());

  edm::ESHandle<HcalConvertedEffectivePedestalsGPU> effectivePedestalsHandle;
  if (configParameters_.useEffectivePedestals)
    effectivePedestalsHandle = setup.getHandle(effectivePedestalsToken_);
  auto const* effectivePedestalsProduct =
      configParameters_.useEffectivePedestals ? &effectivePedestalsHandle->getProduct(ctx.stream()) : nullptr;

  auto const& qieCodersProduct = setup.getData(qieCodersToken_).getProduct(ctx.stream());

  auto const& respCorrsProduct = setup.getData(respCorrsToken_).getProduct(ctx.stream());

  auto const& timeCorrsProduct = setup.getData(timeCorrsToken_).getProduct(ctx.stream());

  auto const& qieTypesProduct = setup.getData(qieTypesToken_).getProduct(ctx.stream());

  HcalTopology const& topology = setup.getData(topologyToken_);
  HcalDDDRecConstants const& recConstants = setup.getData(recConstantsToken_);

  auto const& sipmParametersProduct = setup.getData(sipmParametersToken_).getProduct(ctx.stream());

  auto const& sipmCharacteristicsProduct = setup.getData(sipmCharacteristicsToken_).getProduct(ctx.stream());

  auto const& chQualProduct = setup.getData(chQualProductToken_).getProduct(ctx.stream());

  auto const& pulseOffsets = setup.getData(pulseOffsetsToken_);
  auto const& pulseOffsetsProduct = pulseOffsets.getProduct(ctx.stream());

  // bundle up conditions
  hcal::reconstruction::ConditionsProducts conditions{gainWidthsProduct,
                                                      gainsProduct,
                                                      lutCorrsProduct,
                                                      pedestalWidthsProduct,
                                                      effectivePedestalWidthsProduct,
                                                      pedestalsProduct,
                                                      qieCodersProduct,
                                                      chQualProduct,
                                                      recoParamsProduct,
                                                      respCorrsProduct,
                                                      timeCorrsProduct,
                                                      qieTypesProduct,
                                                      sipmParametersProduct,
                                                      sipmCharacteristicsProduct,
                                                      effectivePedestalsProduct,
                                                      &topology,
                                                      &recConstants,
                                                      pedestals.offsetForHashes(),
                                                      pulseOffsetsProduct,
                                                      pulseOffsets.getValues()};

  // scratch mem on device
  hcal::reconstruction::ScratchDataGPU scratchGPU = {
      cms::cuda::make_device_unique<float[]>(totalChannels * configParameters_.maxTimeSamples, ctx.stream()),
      cms::cuda::make_device_unique<float[]>(totalChannels * configParameters_.maxTimeSamples, ctx.stream()),
      cms::cuda::make_device_unique<float[]>(totalChannels * configParameters_.maxTimeSamples, ctx.stream()),
      cms::cuda::make_device_unique<float[]>(
          totalChannels * configParameters_.maxTimeSamples * configParameters_.maxTimeSamples, ctx.stream()),
      cms::cuda::make_device_unique<float[]>(
          totalChannels * configParameters_.maxTimeSamples * configParameters_.maxTimeSamples, ctx.stream()),
      cms::cuda::make_device_unique<float[]>(
          totalChannels * configParameters_.maxTimeSamples * configParameters_.maxTimeSamples, ctx.stream()),
      cms::cuda::make_device_unique<int8_t[]>(totalChannels, ctx.stream()),
  };

  // output dev mem
  outputGPU_.allocate(configParameters_, totalChannels, ctx.stream());

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

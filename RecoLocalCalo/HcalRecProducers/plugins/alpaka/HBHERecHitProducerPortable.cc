#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

#include "CondFormats/DataRecord/interface/HcalSiPMCharacteristicsRcd.h"
#include "CondFormats/HcalObjects/interface/alpaka/HcalSiPMCharacteristicsDevice.h"
#include "CondFormats/DataRecord/interface/HcalMahiConditionsRcd.h"
#include "CondFormats/HcalObjects/interface/alpaka/HcalMahiConditionsDevice.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "CondFormats/HcalObjects/interface/alpaka/HcalRecoParamWithPulseShapeDevice.h"

#include "DataFormats/HcalDigi/interface/alpaka/HcalDigiDeviceCollection.h"
#include "DataFormats/HcalRecHit/interface/alpaka/HcalRecHitDeviceCollection.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"

#include "HcalMahiPulseOffsetsSoA.h"
#include "Mahi.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace {
    using HcalMahiPulseOffsetsCache =
        cms::alpakatools::MoveToDeviceCache<Device, PortableHostCollection<hcal::HcalMahiPulseOffsetsSoA>>;
  }

  class HBHERecHitProducerPortable : public stream::EDProducer<edm::GlobalCache<HcalMahiPulseOffsetsCache>> {
  public:
    explicit HBHERecHitProducerPortable(edm::ParameterSet const&, HcalMahiPulseOffsetsCache const*);
    ~HBHERecHitProducerPortable() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions&);
    static std::unique_ptr<HcalMahiPulseOffsetsCache> initializeGlobalCache(edm::ParameterSet const& ps);

    static void globalEndJob(HcalMahiPulseOffsetsCache*) {}

  private:
    void produce(device::Event&, device::EventSetup const&) override;

    using IProductTypef01 = hcal::Phase1DigiDeviceCollection;
    const device::EDGetToken<IProductTypef01> digisTokenF01HE_;

    using IProductTypef5 = hcal::Phase0DigiDeviceCollection;
    const device::EDGetToken<IProductTypef5> digisTokenF5HB_;

    using IProductTypef3 = hcal::Phase1DigiDeviceCollection;
    const device::EDGetToken<IProductTypef3> digisTokenF3HB_;

    using OProductType = hcal::RecHitDeviceCollection;
    const device::EDPutToken<OProductType> rechitsM0Token_;

    const device::ESGetToken<hcal::HcalMahiConditionsPortableDevice, HcalMahiConditionsRcd> mahiConditionsToken_;
    const device::ESGetToken<hcal::HcalSiPMCharacteristicsPortableDevice, HcalSiPMCharacteristicsRcd>
        sipmCharacteristicsToken_;
    const device::ESGetToken<hcal::HcalRecoParamWithPulseShapeDevice, HcalRecoParamsRcd> recoParamsToken_;
    //

    hcal::reconstruction::ConfigParameters configParameters_;
  };

  HBHERecHitProducerPortable::HBHERecHitProducerPortable(edm::ParameterSet const& ps, HcalMahiPulseOffsetsCache const*)
      : digisTokenF01HE_{consumes(ps.getParameter<edm::InputTag>("digisLabelF01HE"))},
        digisTokenF5HB_{consumes(ps.getParameter<edm::InputTag>("digisLabelF5HB"))},
        digisTokenF3HB_{consumes(ps.getParameter<edm::InputTag>("digisLabelF3HB"))},
        rechitsM0Token_{produces()},
        mahiConditionsToken_{esConsumes()},
        sipmCharacteristicsToken_{esConsumes()},
        recoParamsToken_{esConsumes()} {
    configParameters_.maxTimeSamples = ps.getParameter<uint32_t>("maxTimeSamples");
    configParameters_.kprep1dChannelsPerBlock = ps.getParameter<uint32_t>("kprep1dChannelsPerBlock");
    configParameters_.sipmQTSShift = ps.getParameter<int>("sipmQTSShift");
    configParameters_.sipmQNTStoSum = ps.getParameter<int>("sipmQNTStoSum");
    configParameters_.firstSampleShift = ps.getParameter<int>("firstSampleShift");
    //TODO: produce only pedestals_width or convertedPedestalWidths depending on this bool
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

  void HBHERecHitProducerPortable::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
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

    desc.add<std::vector<int>>("pulseOffsets", {-3, -2, -1, 0, 1, 2, 3, 4});

    cdesc.addWithDefaultLabel(desc);
  }

  std::unique_ptr<HcalMahiPulseOffsetsCache> HBHERecHitProducerPortable::initializeGlobalCache(
      edm::ParameterSet const& ps) {
    std::vector<int> offsets = ps.getParameter<std::vector<int>>("pulseOffsets");

    PortableHostCollection<hcal::HcalMahiPulseOffsetsSoA> obj(offsets.size(), cms::alpakatools::host());
    auto view = obj.view();

    for (uint32_t i = 0; i < offsets.size(); i++) {
      view[i] = offsets[i];
    }

    return std::make_unique<HcalMahiPulseOffsetsCache>(std::move(obj));
  }

  void HBHERecHitProducerPortable::produce(device::Event& event, device::EventSetup const& setup) {
    auto& queue = event.queue();

    // get device collections from event
    auto const& f01HEDigisDev = event.get(digisTokenF01HE_);
    auto const& f5HBDigisDev = event.get(digisTokenF5HB_);
    auto const& f3HBDigisDev = event.get(digisTokenF3HB_);

    auto const f01DigisSize = f01HEDigisDev->metadata().size();
    auto const f5DigisSize = f5HBDigisDev->metadata().size();
    auto const f3DigisSize = f3HBDigisDev->metadata().size();

    auto const totalChannels = f01DigisSize + f5DigisSize + f3DigisSize;
    OProductType outputGPU_{totalChannels, queue};

    if (totalChannels > 0) {
      // conditions
      auto const& mahiConditionsDev = setup.getData(mahiConditionsToken_);
      auto const& sipmCharacteristicsDev = setup.getData(sipmCharacteristicsToken_);
      auto const& recoParamsWithPulseShapeDev = setup.getData(recoParamsToken_);
      auto const& mahiPulseOffsetsDev = globalCache()->get(queue);

      //
      // schedule algorithms
      //
      hcal::reconstruction::runMahiAsync(queue,
                                         f01HEDigisDev.const_view(),
                                         f5HBDigisDev.const_view(),
                                         f3HBDigisDev.const_view(),
                                         outputGPU_.view(),
                                         mahiConditionsDev.const_view(),
                                         sipmCharacteristicsDev.const_view(),
                                         recoParamsWithPulseShapeDev.const_view(),
                                         mahiPulseOffsetsDev.const_view(),
                                         configParameters_);
    }
    //put into the event
    event.emplace(rechitsM0Token_, std::move(outputGPU_));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HBHERecHitProducerPortable);

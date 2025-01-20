#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "CondFormats/DataRecord/interface/EcalRecHitConditionsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "CondFormats/EcalObjects/interface/EcalRecHitParameters.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitConditionsDevice.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/RecoTypes.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalRecHitDeviceCollection.h"
#include "DataFormats/Portable/interface/PortableObject.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include "DeclsForKernels.h"
#include "EcalRecHitBuilder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace {
    using EcalRecHitParametersCache =
        cms::alpakatools::MoveToDeviceCache<Device, PortableHostObject<EcalRecHitParameters>>;
  }

  class EcalRecHitProducerPortable : public stream::EDProducer<edm::GlobalCache<EcalRecHitParametersCache>> {
  public:
    explicit EcalRecHitProducerPortable(edm::ParameterSet const& ps, EcalRecHitParametersCache const*);
    ~EcalRecHitProducerPortable() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions&);
    static std::unique_ptr<EcalRecHitParametersCache> initializeGlobalCache(edm::ParameterSet const& ps);

    void produce(device::Event&, device::EventSetup const&) override;

    static void globalEndJob(EcalRecHitParametersCache*) {}

  private:
    bool const isPhase2_;
    // input
    using InputProduct = EcalUncalibratedRecHitDeviceCollection;
    const device::EDGetToken<InputProduct> uncalibRecHitsTokenEB_;
    const device::EDGetToken<InputProduct> uncalibRecHitsTokenEE_;
    // output
    using OutputProduct = EcalRecHitDeviceCollection;
    const device::EDPutToken<OutputProduct> recHitsTokenEB_;
    device::EDPutToken<OutputProduct> recHitsTokenEE_;

    // configuration parameters
    ecal::rechit::ConfigurationParameters configParameters_;

    // conditions tokens
    const device::ESGetToken<EcalRecHitConditionsDevice, EcalRecHitConditionsRcd> recHitConditionsToken_;
  };

  void EcalRecHitProducerPortable::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("uncalibrecHitsInLabelEB",
                            edm::InputTag("ecalMultiFitUncalibRecHitPortable", "EcalUncalibRecHitsEB"));
    desc.add<std::string>("recHitsLabelEB", "EcalRecHitsEB");
    desc.add<bool>("killDeadChannels", true);
    desc.add<bool>("recoverEBIsolatedChannels", false);
    desc.add<bool>("recoverEBVFE", false);
    desc.add<bool>("recoverEBFE", true);

    desc.add<double>("EBLaserMIN", 0.5);
    desc.add<double>("EBLaserMAX", 3.0);

    desc.ifValue(edm::ParameterDescription<bool>("isPhase2", false, true),
                 false >> (edm::ParameterDescription<edm::InputTag>(
                               "uncalibrecHitsInLabelEE",
                               edm::InputTag("ecalMultiFitUncalibRecHitPortable", "EcalUncalibRecHitsEE"),
                               true) and
                           edm::ParameterDescription<std::string>("recHitsLabelEE", "EcalRecHitsEE", true) and
                           edm::ParameterDescription<bool>("recoverEEIsolatedChannels", false, true) and
                           edm::ParameterDescription<bool>("recoverEEVFE", false, true) and
                           edm::ParameterDescription<bool>("recoverEEFE", true, true) and
                           edm::ParameterDescription<double>("EELaserMIN", 0.5, true) and
                           edm::ParameterDescription<double>("EELaserMAX", 8.0, true)) or
                     true >> edm::EmptyGroupDescription());

    // channel statuses to be exluded from reconstruction
    desc.add<std::vector<std::string>>("ChannelStatusToBeExcluded",
                                       {"kDAC",
                                        "kNoisy",
                                        "kNNoisy",
                                        "kFixedG6",
                                        "kFixedG1",
                                        "kFixedG0",
                                        "kNonRespondingIsolated",
                                        "kDeadVFE",
                                        "kDeadFE",
                                        "kNoDataNoTP"});

    // reco flags association to channel status flags
    edm::ParameterSetDescription psd0;
    psd0.add<std::vector<std::string>>("kGood", {"kOk", "kDAC", "kNoLaser", "kNoisy"});
    psd0.add<std::vector<std::string>>("kNeighboursRecovered", {"kFixedG0", "kNonRespondingIsolated", "kDeadVFE"});
    psd0.add<std::vector<std::string>>("kDead", {"kNoDataNoTP"});
    psd0.add<std::vector<std::string>>("kNoisy", {"kNNoisy", "kFixedG6", "kFixedG1"});
    psd0.add<std::vector<std::string>>("kTowerRecovered", {"kDeadFE"});
    desc.add<edm::ParameterSetDescription>("flagsMapDBReco", psd0);

    confDesc.addWithDefaultLabel(desc);
  }

  std::unique_ptr<EcalRecHitParametersCache> EcalRecHitProducerPortable::initializeGlobalCache(
      edm::ParameterSet const& ps) {
    PortableHostObject<EcalRecHitParameters> params(cms::alpakatools::host());

    // Translate string representation of ChannelStatusToBeExcluded to enum values and pack into bitset
    auto const& channelStatusToBeExcluded = StringToEnumValue<EcalChannelStatusCode::Code>(
        ps.getParameter<std::vector<std::string>>("ChannelStatusToBeExcluded"));
    for (auto const& st : channelStatusToBeExcluded) {
      params->channelStatusCodesToBeExcluded.set(st);
    }

    // Generate map of channel status codes and corresponding recoFlag bits
    auto const& fmdbRecoPset = ps.getParameter<edm::ParameterSet>("flagsMapDBReco");
    auto const& recoFlagStrings = fmdbRecoPset.getParameterNames();
    for (auto const& recoFlagString : recoFlagStrings) {
      auto const recoFlag = static_cast<EcalRecHit::Flags>(StringToEnumValue<EcalRecHit::Flags>(recoFlagString));
      auto const& channelStatusCodeStrings = fmdbRecoPset.getParameter<std::vector<std::string>>(recoFlagString);
      for (auto const& channelStatusCodeString : channelStatusCodeStrings) {
        auto const chStatCode = StringToEnumValue<EcalChannelStatusCode::Code>(channelStatusCodeString);
        // set recoFlagBits for this channel status code
        params->recoFlagBits.at(chStatCode) = static_cast<uint32_t>(recoFlag);
      }
    }

    return std::make_unique<EcalRecHitParametersCache>(std::move(params));
  }

  EcalRecHitProducerPortable::EcalRecHitProducerPortable(const edm::ParameterSet& ps, EcalRecHitParametersCache const*)
      : isPhase2_{ps.getParameter<bool>("isPhase2")},
        uncalibRecHitsTokenEB_{consumes(ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEB"))},
        uncalibRecHitsTokenEE_{isPhase2_ ? device::EDGetToken<InputProduct>{}
                                         : consumes(ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEE"))},
        recHitsTokenEB_{produces(ps.getParameter<std::string>("recHitsLabelEB"))},
        recHitConditionsToken_{esConsumes()} {
    if (!isPhase2_) {
      recHitsTokenEE_ = produces(ps.getParameter<std::string>("recHitsLabelEE"));
    }
    configParameters_.killDeadChannels = ps.getParameter<bool>("killDeadChannels");
    configParameters_.EBLaserMIN = ps.getParameter<double>("EBLaserMIN");
    configParameters_.EELaserMIN = isPhase2_ ? 0. : ps.getParameter<double>("EELaserMIN");
    configParameters_.EBLaserMAX = ps.getParameter<double>("EBLaserMAX");
    configParameters_.EELaserMAX = isPhase2_ ? 0. : ps.getParameter<double>("EELaserMAX");

    // do not propagate channels with these flags on
    uint32_t flagmask = 0;
    flagmask |= 0x1 << EcalRecHit::kNeighboursRecovered;
    flagmask |= 0x1 << EcalRecHit::kTowerRecovered;
    flagmask |= 0x1 << EcalRecHit::kDead;
    flagmask |= 0x1 << EcalRecHit::kKilled;
    flagmask |= 0x1 << EcalRecHit::kTPSaturated;
    flagmask |= 0x1 << EcalRecHit::kL1SpikeFlag;
    configParameters_.flagmask = flagmask;

    // for recovery and killing
    configParameters_.recoverEBIsolatedChannels = ps.getParameter<bool>("recoverEBIsolatedChannels");
    configParameters_.recoverEEIsolatedChannels =
        isPhase2_ ? false : ps.getParameter<bool>("recoverEEIsolatedChannels");
    configParameters_.recoverEBVFE = ps.getParameter<bool>("recoverEBVFE");
    configParameters_.recoverEEVFE = isPhase2_ ? false : ps.getParameter<bool>("recoverEEVFE");
    configParameters_.recoverEBFE = ps.getParameter<bool>("recoverEBFE");
    configParameters_.recoverEEFE = isPhase2_ ? false : ps.getParameter<bool>("recoverEEFE");
  }

  void EcalRecHitProducerPortable::produce(device::Event& event, device::EventSetup const& setup) {
    auto& queue = event.queue();

    // get device collections from event
    auto const* uncalibRecHitsDevEB = &event.get(uncalibRecHitsTokenEB_);
    auto const* uncalibRecHitsDevEE = isPhase2_ ? nullptr : &event.get(uncalibRecHitsTokenEE_);

    // get the size of the input collections from the metadata
    auto const uncalibRecHitsSizeEB = uncalibRecHitsDevEB->const_view().metadata().size();
    auto const uncalibRecHitsSizeEE = isPhase2_ ? 0 : uncalibRecHitsDevEE->const_view().metadata().size();

    // output device collections with the same size than the input collections
    auto recHitsDevEB = std::make_unique<OutputProduct>(uncalibRecHitsSizeEB, queue);
    auto recHitsDevEE =
        isPhase2_ ? std::unique_ptr<OutputProduct>() : std::make_unique<OutputProduct>(uncalibRecHitsSizeEE, queue);
    // reset the size scalar of the SoA
    // memset takes an alpaka view that is created from the scalar in a view to the portable device collection
    auto recHitSizeViewEB = cms::alpakatools::make_device_view<uint32_t>(queue, recHitsDevEB->view().size());
    alpaka::memset(queue, recHitSizeViewEB, 0);

    if (!isPhase2_) {
      auto recHitSizeViewEE = cms::alpakatools::make_device_view<uint32_t>(queue, recHitsDevEE->view().size());
      alpaka::memset(queue, recHitSizeViewEE, 0);
    }

    // stop here if there are no uncalibrated rechits
    if (uncalibRecHitsSizeEB + uncalibRecHitsSizeEE > 0) {
      // to get the event time from device::Event one has to access the underlying edm::Event
      auto const& eventTime = static_cast<const edm::Event&>(event).time().value();

      // conditions
      auto const& recHitConditionsDev = setup.getData(recHitConditionsToken_);
      auto const* recHitParametersDev = globalCache()->get(queue).const_data();

      //
      // schedule algorithms
      //
      ecal::rechit::create_ecal_rechit(queue,
                                       uncalibRecHitsDevEB,
                                       uncalibRecHitsDevEE,
                                       *recHitsDevEB,
                                       *recHitsDevEE,
                                       recHitConditionsDev,
                                       recHitParametersDev,
                                       eventTime,
                                       configParameters_,
                                       isPhase2_);
    }

    // put collections into the event
    event.put(recHitsTokenEB_, std::move(recHitsDevEB));
    if (!isPhase2_) {
      event.put(recHitsTokenEE_, std::move(recHitsDevEE));
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(EcalRecHitProducerPortable);

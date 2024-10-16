#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/EcalRecHitParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "CondFormats/EcalObjects/interface/EcalRecHitParametersSoA.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitParametersDevice.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class EcalRecHitParametersHostESProducer : public ESProducer {
  public:
    EcalRecHitParametersHostESProducer(edm::ParameterSet const&);
    ~EcalRecHitParametersHostESProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions&);
    std::unique_ptr<EcalRecHitParametersHost> produce(EcalRecHitParametersRcd const&);

  private:
    std::bitset<kNEcalChannelStatusCodes> channelStatusCodesToBeExcluded_;
    RecoFlagBitsArray recoFlagBitsArray_;
  };

  EcalRecHitParametersHostESProducer::EcalRecHitParametersHostESProducer(edm::ParameterSet const& iConfig)
      : ESProducer(iConfig), recoFlagBitsArray_() {
    setWhatProduced(this);

    // Translate string representation of ChannelStatusToBeExcluded to enum values and pack into bitset
    auto const& channelStatusToBeExcluded = StringToEnumValue<EcalChannelStatusCode::Code>(
        iConfig.getParameter<std::vector<std::string>>("ChannelStatusToBeExcluded"));
    for (auto const& st : channelStatusToBeExcluded) {
      channelStatusCodesToBeExcluded_.set(st);
    }

    // Generate map of channel status codes and corresponding recoFlag bits
    auto const& fmdbRecoPset = iConfig.getParameter<edm::ParameterSet>("flagsMapDBReco");
    auto const& recoFlagStrings = fmdbRecoPset.getParameterNames();
    for (auto const& recoFlagString : recoFlagStrings) {
      auto const recoFlag = static_cast<EcalRecHit::Flags>(StringToEnumValue<EcalRecHit::Flags>(recoFlagString));
      auto const& channelStatusCodeStrings = fmdbRecoPset.getParameter<std::vector<std::string>>(recoFlagString);
      for (auto const& channelStatusCodeString : channelStatusCodeStrings) {
        auto const chStatCode = StringToEnumValue<EcalChannelStatusCode::Code>(channelStatusCodeString);
        // set recoFlagBits for this channel status code
        recoFlagBitsArray_.at(chStatCode) = static_cast<uint32_t>(recoFlag);
      }
    }
  }

  void EcalRecHitParametersHostESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
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

    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<EcalRecHitParametersHost> EcalRecHitParametersHostESProducer::produce(
      EcalRecHitParametersRcd const& iRecord) {
    size_t const sizeone = 1;
    auto product = std::make_unique<EcalRecHitParametersHost>(sizeone, cms::alpakatools::host());
    auto view = product->view();

    std::memcpy(view.recoFlagBits().data(), recoFlagBitsArray_.data(), sizeof(uint32_t) * recoFlagBitsArray_.size());

    view.channelStatusCodesToBeExcluded() = channelStatusCodesToBeExcluded_;

    return product;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(EcalRecHitParametersHostESProducer);

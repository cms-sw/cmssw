#include "CondFormats/DataRecord/interface/EcalRecHitConditionsRcd.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitConditionsDevice.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitParametersDevice.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/RecoTypes.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalRecHitDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"

#include "DeclsForKernels.h"
#include "EcalRecHitBuilder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class EcalRecHitProducerPortable : public stream::EDProducer<> {
  public:
    explicit EcalRecHitProducerPortable(edm::ParameterSet const& ps);
    ~EcalRecHitProducerPortable() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

    void produce(device::Event&, device::EventSetup const&) override;

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
    const device::ESGetToken<EcalRecHitParametersDevice, JobConfigurationGPURecord> recHitParametersToken_;
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

    confDesc.addWithDefaultLabel(desc);
  }

  EcalRecHitProducerPortable::EcalRecHitProducerPortable(const edm::ParameterSet& ps)
      : isPhase2_{ps.getParameter<bool>("isPhase2")},
        uncalibRecHitsTokenEB_{consumes(ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEB"))},
        uncalibRecHitsTokenEE_{isPhase2_ ? device::EDGetToken<InputProduct>{}
                                         : consumes(ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEE"))},
        recHitsTokenEB_{produces(ps.getParameter<std::string>("recHitsLabelEB"))},
        recHitConditionsToken_{esConsumes()},
        recHitParametersToken_{esConsumes()} {
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
      auto const& recHitParametersDev = setup.getData(recHitParametersToken_);

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

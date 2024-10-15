#include "CondFormats/DataRecord/interface/EcalRecHitConditionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalRecHitParametersRcd.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitConditionsDevice.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitParametersDevice.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/RecoTypes.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalRecHitDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

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
    // input
    using InputProduct = EcalUncalibratedRecHitDeviceCollection;
    const device::EDGetToken<InputProduct> uncalibRecHitsTokenEB_;
    const device::EDGetToken<InputProduct> uncalibRecHitsTokenEE_;
    // output
    using OutputProduct = EcalRecHitDeviceCollection;
    const device::EDPutToken<OutputProduct> recHitsTokenEB_;
    const device::EDPutToken<OutputProduct> recHitsTokenEE_;

    // configuration parameters
    ecal::rechit::ConfigurationParameters configParameters_;

    // conditions tokens
    const device::ESGetToken<EcalRecHitConditionsDevice, EcalRecHitConditionsRcd> recHitConditionsToken_;
    const device::ESGetToken<EcalRecHitParametersDevice, EcalRecHitParametersRcd> recHitParametersToken_;
  };

  void EcalRecHitProducerPortable::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("uncalibrecHitsInLabelEB",
                            edm::InputTag("ecalMultiFitUncalibRecHitPortable", "EcalUncalibRecHitsEB"));
    desc.add<edm::InputTag>("uncalibrecHitsInLabelEE",
                            edm::InputTag("ecalMultiFitUncalibRecHitPortable", "EcalUncalibRecHitsEE"));

    desc.add<std::string>("recHitsLabelEB", "EcalRecHitsEB");
    desc.add<std::string>("recHitsLabelEE", "EcalRecHitsEE");

    desc.add<bool>("killDeadChannels", true);
    desc.add<bool>("recoverEBIsolatedChannels", false);
    desc.add<bool>("recoverEEIsolatedChannels", false);
    desc.add<bool>("recoverEBVFE", false);
    desc.add<bool>("recoverEEVFE", false);
    desc.add<bool>("recoverEBFE", true);
    desc.add<bool>("recoverEEFE", true);

    desc.add<double>("EBLaserMIN", 0.5);
    desc.add<double>("EELaserMIN", 0.5);
    desc.add<double>("EBLaserMAX", 3.0);
    desc.add<double>("EELaserMAX", 8.0);

    confDesc.addWithDefaultLabel(desc);
  }

  EcalRecHitProducerPortable::EcalRecHitProducerPortable(const edm::ParameterSet& ps)
      : uncalibRecHitsTokenEB_{consumes(ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEB"))},
        uncalibRecHitsTokenEE_{consumes(ps.getParameter<edm::InputTag>("uncalibrecHitsInLabelEE"))},
        recHitsTokenEB_{produces(ps.getParameter<std::string>("recHitsLabelEB"))},
        recHitsTokenEE_{produces(ps.getParameter<std::string>("recHitsLabelEE"))},
        recHitConditionsToken_{esConsumes()},
        recHitParametersToken_{esConsumes()} {
    configParameters_.killDeadChannels = ps.getParameter<bool>("killDeadChannels");
    configParameters_.EBLaserMIN = ps.getParameter<double>("EBLaserMIN");
    configParameters_.EELaserMIN = ps.getParameter<double>("EELaserMIN");
    configParameters_.EBLaserMAX = ps.getParameter<double>("EBLaserMAX");
    configParameters_.EELaserMAX = ps.getParameter<double>("EELaserMAX");

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
    configParameters_.recoverEEIsolatedChannels = ps.getParameter<bool>("recoverEEIsolatedChannels");
    configParameters_.recoverEBVFE = ps.getParameter<bool>("recoverEBVFE");
    configParameters_.recoverEEVFE = ps.getParameter<bool>("recoverEEVFE");
    configParameters_.recoverEBFE = ps.getParameter<bool>("recoverEBFE");
    configParameters_.recoverEEFE = ps.getParameter<bool>("recoverEEFE");
  }

  //void EcalRecHitProducerPortable::acquire(device::Event const& event, device::EventSetup const& setup) {
  //  auto& queue = event.queue();

  //  // get device collections from event
  //  auto const& ebUncalibRecHitsDev = event.get(uncalibRecHitsEBToken_);
  //  auto const& eeUncalibRecHitsDev = event.get(uncalibRecHitsEEToken_);

  //  // copy the actual numbers of uncalibrated rechits in the collections to host
  //  auto ebUncalibratedRecHitsSizeDevConstView =
  //      cms::alpakatools::make_device_view<const uint32_t>(alpaka::getDev(queue), ebUncalibRecHitsDev.const_view().size());
  //  auto eeUncalibratedRecHitsSizeDevConstView =
  //      cms::alpakatools::make_device_view<const uint32_t>(alpaka::getDev(queue), eeUncalibRecHitsDev.const_view().size());
  //  alpaka::memcpy(queue, ebUncalibratedRecHitsSizeHostBuf_, ebUncalibratedRecHitsSizeDevConstView);
  //  alpaka::memcpy(queue, eeUncalibratedRecHitsSizeHostBuf_, eeUncalibratedRecHitsSizeDevConstView);

  //neb_ = ebUncalibRecHits.size;
  //nee_ = eeUncalibRecHits.size;

  //// stop here if there are no uncalibRecHits
  //if (neb_ + nee_ == 0)
  //  return;

  //int nchannelsEB = ebUncalibRecHits.size;  // --> offsetForInput, first EB and then EE

  //// conditions
  //// - laser correction
  //// - IC
  //// - adt2gev

  ////
  //IntercalibConstantsHandle_ = setup.getHandle(tokenIntercalibConstants_);
  //recHitParametersHandle_ = setup.getHandle(tokenRecHitParameters_);

  //auto const& ADCToGeVConstantProduct = setup.getData(tokenADCToGeVConstant_).getProduct(ctx.stream());
  //auto const& IntercalibConstantsProduct = IntercalibConstantsHandle_->getProduct(ctx.stream());
  //auto const& ChannelStatusProduct = setup.getData(tokenChannelStatus_).getProduct(ctx.stream());

  //auto const& LaserAPDPNRatiosProduct = setup.getData(tokenLaserAPDPNRatios_).getProduct(ctx.stream());
  //auto const& LaserAPDPNRatiosRefProduct = setup.getData(tokenLaserAPDPNRatiosRef_).getProduct(ctx.stream());
  //auto const& LaserAlphasProduct = setup.getData(tokenLaserAlphas_).getProduct(ctx.stream());
  //auto const& LinearCorrectionsProduct = setup.getData(tokenLinearCorrections_).getProduct(ctx.stream());
  //auto const& recHitParametersProduct = recHitParametersHandle_->getProduct(ctx.stream());

  //// set config ptrs : this is done to avoid changing things downstream
  //configParameters_.ChannelStatusToBeExcluded = recHitParametersProduct.channelStatusToBeExcluded.get();
  //configParameters_.ChannelStatusToBeExcludedSize = std::get<0>(recHitParametersHandle_->getValues()).get().size();
  //configParameters_.expanded_v_DB_reco_flags = recHitParametersProduct.expanded_v_DB_reco_flags.get();
  //configParameters_.expanded_Sizes_v_DB_reco_flags = recHitParametersProduct.expanded_Sizes_v_DB_reco_flags.get();
  //configParameters_.expanded_flagbit_v_DB_reco_flags = recHitParametersProduct.expanded_flagbit_v_DB_reco_flags.get();
  //configParameters_.expanded_v_DB_reco_flagsSize = std::get<3>(recHitParametersHandle_->getValues()).get().size();

  //// bundle up conditions
  //ecal::rechit::ConditionsProducts conditions{ADCToGeVConstantProduct,
  //                                            IntercalibConstantsProduct,
  //                                            ChannelStatusProduct,
  //                                            LaserAPDPNRatiosProduct,
  //                                            LaserAPDPNRatiosRefProduct,
  //                                            LaserAlphasProduct,
  //                                            LinearCorrectionsProduct,
  //                                            IntercalibConstantsHandle_->getOffset()};

  //// dev mem
  //eventOutputDataGPU_.allocate(configParameters_, neb_, nee_, ctx.stream());

  ////
  //// schedule algorithms
  ////

  //edm::TimeValue_t event_time = event.time().value();

  //ecal::rechit::create_ecal_rehit(
  //    inputDataGPU, eventOutputDataGPU_, conditions, configParameters_, nchannelsEB, event_time, ctx.stream());

  //cudaCheck(cudaGetLastError());
  //}

  void EcalRecHitProducerPortable::produce(device::Event& event, device::EventSetup const& setup) {
    auto& queue = event.queue();

    // get device collections from event
    auto const& uncalibRecHitsDevEB = event.get(uncalibRecHitsTokenEB_);
    auto const& uncalibRecHitsDevEE = event.get(uncalibRecHitsTokenEE_);

    // get the size of the input collections from the metadata
    auto const uncalibRecHitsSizeEB = uncalibRecHitsDevEB.const_view().metadata().size();
    auto const uncalibRecHitsSizeEE = uncalibRecHitsDevEE.const_view().metadata().size();

    // output device collections with the same size than the input collections
    OutputProduct recHitsDevEB{uncalibRecHitsSizeEB, queue};
    OutputProduct recHitsDevEE{uncalibRecHitsSizeEE, queue};
    // reset the size scalar of the SoA
    // memset takes an alpaka view that is created from the scalar in a view to the portable device collection
    auto recHitSizeViewEB =
        cms::alpakatools::make_device_view<uint32_t>(alpaka::getDev(queue), recHitsDevEB.view().size());
    auto recHitSizeViewEE =
        cms::alpakatools::make_device_view<uint32_t>(alpaka::getDev(queue), recHitsDevEE.view().size());
    alpaka::memset(queue, recHitSizeViewEB, 0);
    alpaka::memset(queue, recHitSizeViewEE, 0);

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
                                       recHitsDevEB,
                                       recHitsDevEE,
                                       recHitConditionsDev,
                                       recHitParametersDev,
                                       eventTime,
                                       configParameters_);
    }

    // put collections into the event
    event.emplace(recHitsTokenEB_, std::move(recHitsDevEB));
    event.emplace(recHitsTokenEE_, std::move(recHitsDevEE));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(EcalRecHitProducerPortable);

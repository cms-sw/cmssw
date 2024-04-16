#include "CondFormats/DataRecord/interface/EcalMultifitConditionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalMultifitParametersRcd.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitConditionsDevice.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitParametersDevice.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"

#include "DeclsForKernels.h"
#include "EcalUncalibRecHitMultiFitAlgoPortable.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class EcalUncalibRecHitProducerPortable : public stream::SynchronizingEDProducer<> {
  public:
    explicit EcalUncalibRecHitProducerPortable(edm::ParameterSet const& ps);
    ~EcalUncalibRecHitProducerPortable() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

    void acquire(device::Event const&, device::EventSetup const&) override;
    void produce(device::Event&, device::EventSetup const&) override;

  private:
    using InputProduct = EcalDigiDeviceCollection;
    const device::EDGetToken<InputProduct> digisTokenEB_;
    const device::EDGetToken<InputProduct> digisTokenEE_;
    using OutputProduct = EcalUncalibratedRecHitDeviceCollection;
    const device::EDPutToken<OutputProduct> uncalibRecHitsTokenEB_;
    const device::EDPutToken<OutputProduct> uncalibRecHitsTokenEE_;

    // conditions tokens
    const device::ESGetToken<EcalMultifitConditionsDevice, EcalMultifitConditionsRcd> multifitConditionsToken_;
    const device::ESGetToken<EcalMultifitParametersDevice, EcalMultifitParametersRcd> multifitParametersToken_;

    // configuration parameters
    ecal::multifit::ConfigurationParameters configParameters_;

    cms::alpakatools::host_buffer<uint32_t> ebDigisSizeHostBuf_;
    cms::alpakatools::host_buffer<uint32_t> eeDigisSizeHostBuf_;
  };

  void EcalUncalibRecHitProducerPortable::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("digisLabelEB", edm::InputTag("ecalRawToDigiPortable", "ebDigis"));
    desc.add<edm::InputTag>("digisLabelEE", edm::InputTag("ecalRawToDigiPortable", "eeDigis"));

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
    desc.addUntracked<std::vector<uint32_t>>("kernelMinimizeThreads", {32, 1, 1});
    desc.add<bool>("shouldRunTimingComputation", true);
    confDesc.addWithDefaultLabel(desc);
  }

  EcalUncalibRecHitProducerPortable::EcalUncalibRecHitProducerPortable(const edm::ParameterSet& ps)
      : digisTokenEB_{consumes(ps.getParameter<edm::InputTag>("digisLabelEB"))},
        digisTokenEE_{consumes(ps.getParameter<edm::InputTag>("digisLabelEE"))},
        uncalibRecHitsTokenEB_{produces(ps.getParameter<std::string>("recHitsLabelEB"))},
        uncalibRecHitsTokenEE_{produces(ps.getParameter<std::string>("recHitsLabelEE"))},
        multifitConditionsToken_{esConsumes()},
        multifitParametersToken_{esConsumes()},
        ebDigisSizeHostBuf_{cms::alpakatools::make_host_buffer<uint32_t>()},
        eeDigisSizeHostBuf_{cms::alpakatools::make_host_buffer<uint32_t>()} {
    // Workaround until the ProductID problem in issue https://github.com/cms-sw/cmssw/issues/44643 is fixed
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    producesTemporarily("edm::DeviceProduct<alpaka_cuda_async::EcalUncalibratedRecHitDeviceCollection>",
                        ps.getParameter<std::string>("recHitsLabelEB"));
    producesTemporarily("edm::DeviceProduct<alpaka_cuda_async::EcalUncalibratedRecHitDeviceCollection>",
                        ps.getParameter<std::string>("recHitsLabelEE"));
#endif

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

  void EcalUncalibRecHitProducerPortable::acquire(device::Event const& event, device::EventSetup const& setup) {
    auto& queue = event.queue();

    // get device collections from event
    auto const& ebDigisDev = event.get(digisTokenEB_);
    auto const& eeDigisDev = event.get(digisTokenEE_);

    // copy the actual numbers of digis in the collections to host
    auto ebDigisSizeDevConstView =
        cms::alpakatools::make_device_view<const uint32_t>(alpaka::getDev(queue), ebDigisDev.const_view().size());
    auto eeDigisSizeDevConstView =
        cms::alpakatools::make_device_view<const uint32_t>(alpaka::getDev(queue), eeDigisDev.const_view().size());
    alpaka::memcpy(queue, ebDigisSizeHostBuf_, ebDigisSizeDevConstView);
    alpaka::memcpy(queue, eeDigisSizeHostBuf_, eeDigisSizeDevConstView);
  }

  void EcalUncalibRecHitProducerPortable::produce(device::Event& event, device::EventSetup const& setup) {
    auto& queue = event.queue();

    // get device collections from event
    auto const& ebDigisDev = event.get(digisTokenEB_);
    auto const& eeDigisDev = event.get(digisTokenEE_);

    // get the actual numbers of digis in the collections
    auto const ebDigisSize = static_cast<int>(*ebDigisSizeHostBuf_.data());
    auto const eeDigisSize = static_cast<int>(*eeDigisSizeHostBuf_.data());

    // output device collections
    OutputProduct uncalibRecHitsDevEB{ebDigisSize, queue};
    OutputProduct uncalibRecHitsDevEE{eeDigisSize, queue};
    // reset the size scalar of the SoA
    // memset takes an alpaka view that is created from the scalar in a view to the portable device collection
    auto uncalibRecHitSizeViewEB =
        cms::alpakatools::make_device_view<uint32_t>(alpaka::getDev(queue), uncalibRecHitsDevEB.view().size());
    auto uncalibRecHitSizeViewEE =
        cms::alpakatools::make_device_view<uint32_t>(alpaka::getDev(queue), uncalibRecHitsDevEE.view().size());
    alpaka::memset(queue, uncalibRecHitSizeViewEB, 0);
    alpaka::memset(queue, uncalibRecHitSizeViewEE, 0);

    // stop here if there are no digis
    if (ebDigisSize + eeDigisSize > 0) {
      // conditions
      auto const& multifitConditionsDev = setup.getData(multifitConditionsToken_);
      auto const& multifitParametersDev = setup.getData(multifitParametersToken_);

      //
      // schedule algorithms
      //
      ecal::multifit::launchKernels(queue,
                                    ebDigisDev,
                                    eeDigisDev,
                                    uncalibRecHitsDevEB,
                                    uncalibRecHitsDevEE,
                                    multifitConditionsDev,
                                    multifitParametersDev,
                                    configParameters_);
    }

    // put into the event
    event.emplace(uncalibRecHitsTokenEB_, std::move(uncalibRecHitsDevEB));
    event.emplace(uncalibRecHitsTokenEE_, std::move(uncalibRecHitsDevEE));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(EcalUncalibRecHitProducerPortable);

#include "CondFormats/DataRecord/interface/EcalMultifitConditionsRcd.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitConditionsDevice.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "DataFormats/Portable/interface/PortableObject.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"

#include "DeclsForKernels.h"
#include "EcalMultifitParameters.h"
#include "EcalUncalibRecHitMultiFitAlgoPortable.h"

#include <algorithm>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace {
    using EcalMultifitParametersCache =
        cms::alpakatools::MoveToDeviceCache<Device, PortableHostObject<EcalMultifitParameters>>;
  }

  class EcalUncalibRecHitProducerPortable
      : public stream::SynchronizingEDProducer<edm::GlobalCache<EcalMultifitParametersCache>> {
  public:
    explicit EcalUncalibRecHitProducerPortable(edm::ParameterSet const& ps, EcalMultifitParametersCache const*);
    ~EcalUncalibRecHitProducerPortable() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions&);
    static std::unique_ptr<EcalMultifitParametersCache> initializeGlobalCache(edm::ParameterSet const& ps);

    void acquire(device::Event const&, device::EventSetup const&) override;
    void produce(device::Event&, device::EventSetup const&) override;

    static void globalEndJob(EcalMultifitParametersCache*) {}

  private:
    using InputProduct = EcalDigiDeviceCollection;
    const device::EDGetToken<InputProduct> digisTokenEB_;
    const device::EDGetToken<InputProduct> digisTokenEE_;
    using OutputProduct = EcalUncalibratedRecHitDeviceCollection;
    const device::EDPutToken<OutputProduct> uncalibRecHitsTokenEB_;
    const device::EDPutToken<OutputProduct> uncalibRecHitsTokenEE_;

    // conditions tokens
    const device::ESGetToken<EcalMultifitConditionsDevice, EcalMultifitConditionsRcd> multifitConditionsToken_;

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

    desc.add<std::vector<double>>("EBtimeFitParameters",
                                  {-2.015452e+00,
                                   3.130702e+00,
                                   -1.234730e+01,
                                   4.188921e+01,
                                   -8.283944e+01,
                                   9.101147e+01,
                                   -5.035761e+01,
                                   1.105621e+01});
    desc.add<std::vector<double>>("EEtimeFitParameters",
                                  {-2.390548e+00,
                                   3.553628e+00,
                                   -1.762341e+01,
                                   6.767538e+01,
                                   -1.332130e+02,
                                   1.407432e+02,
                                   -7.541106e+01,
                                   1.620277e+01});
    desc.add<std::vector<double>>("EBamplitudeFitParameters", {1.138, 1.652});
    desc.add<std::vector<double>>("EEamplitudeFitParameters", {1.890, 1.400});

    desc.addUntracked<std::vector<uint32_t>>("kernelMinimizeThreads", {32, 1, 1});
    desc.add<bool>("shouldRunTimingComputation", true);

    confDesc.addWithDefaultLabel(desc);
  }

  std::unique_ptr<EcalMultifitParametersCache> EcalUncalibRecHitProducerPortable::initializeGlobalCache(
      edm::ParameterSet const& ps) {
    PortableHostObject<EcalMultifitParameters> params(cms::alpakatools::host());

    auto const ebTimeFitParamsFromPSet = ps.getParameter<std::vector<double>>("EBtimeFitParameters");
    auto const eeTimeFitParamsFromPSet = ps.getParameter<std::vector<double>>("EEtimeFitParameters");
    // Assert that there are as many parameters as the EcalMultiFitParametersSoA expects
    assert(ebTimeFitParamsFromPSet.size() == EcalMultifitParameters::kNTimeFitParams);
    assert(eeTimeFitParamsFromPSet.size() == EcalMultifitParameters::kNTimeFitParams);
    std::ranges::copy(ebTimeFitParamsFromPSet, params->timeFitParamsEB.begin());
    std::ranges::copy(eeTimeFitParamsFromPSet, params->timeFitParamsEE.begin());

    std::vector<float> ebAmplitudeFitParameters_;
    std::vector<float> eeAmplitudeFitParameters_;
    auto const ebAmplFitParamsFromPSet = ps.getParameter<std::vector<double>>("EBamplitudeFitParameters");
    auto const eeAmplFitParamsFromPSet = ps.getParameter<std::vector<double>>("EEamplitudeFitParameters");
    // Assert that there are as many parameters as the EcalMultiFitParametersSoA expects
    assert(ebAmplFitParamsFromPSet.size() == EcalMultifitParameters::kNAmplitudeFitParams);
    assert(eeAmplFitParamsFromPSet.size() == EcalMultifitParameters::kNAmplitudeFitParams);
    std::ranges::copy(ebAmplFitParamsFromPSet, params->amplitudeFitParamsEB.begin());
    std::ranges::copy(eeAmplFitParamsFromPSet, params->amplitudeFitParamsEE.begin());

    return std::make_unique<EcalMultifitParametersCache>(std::move(params));
  }

  EcalUncalibRecHitProducerPortable::EcalUncalibRecHitProducerPortable(const edm::ParameterSet& ps,
                                                                       EcalMultifitParametersCache const*)
      : SynchronizingEDProducer(ps),
        digisTokenEB_{consumes(ps.getParameter<edm::InputTag>("digisLabelEB"))},
        digisTokenEE_{consumes(ps.getParameter<edm::InputTag>("digisLabelEE"))},
        uncalibRecHitsTokenEB_{produces(ps.getParameter<std::string>("recHitsLabelEB"))},
        uncalibRecHitsTokenEE_{produces(ps.getParameter<std::string>("recHitsLabelEE"))},
        multifitConditionsToken_{esConsumes()},
        ebDigisSizeHostBuf_{cms::alpakatools::make_host_buffer<uint32_t>()},
        eeDigisSizeHostBuf_{cms::alpakatools::make_host_buffer<uint32_t>()} {
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
        cms::alpakatools::make_device_view<const uint32_t>(queue, ebDigisDev.const_view().size());
    auto eeDigisSizeDevConstView =
        cms::alpakatools::make_device_view<const uint32_t>(queue, eeDigisDev.const_view().size());
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
        cms::alpakatools::make_device_view<uint32_t>(queue, uncalibRecHitsDevEB.view().size());
    auto uncalibRecHitSizeViewEE =
        cms::alpakatools::make_device_view<uint32_t>(queue, uncalibRecHitsDevEE.view().size());
    alpaka::memset(queue, uncalibRecHitSizeViewEB, 0);
    alpaka::memset(queue, uncalibRecHitSizeViewEE, 0);

    // stop here if there are no digis
    if (ebDigisSize + eeDigisSize > 0) {
      // conditions
      auto const& multifitConditionsDev = setup.getData(multifitConditionsToken_);
      auto const* multifitParametersDev = globalCache()->get(queue).const_data();

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

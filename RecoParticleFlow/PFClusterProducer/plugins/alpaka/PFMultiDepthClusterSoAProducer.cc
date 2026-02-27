#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Portable/interface/PortableObject.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterParams.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizer.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    using PFMultiDepthClusterParamsCache =
        cms::alpakatools::MoveToDeviceCache<Device, PortableHostObject<PFMultiDepthClusterParams>>;
  }

  class PFMultiDepthClusterSoAProducer
      : public stream::SynchronizingEDProducer<edm::GlobalCache<PFMultiDepthClusterParamsCache>> {
  public:
    PFMultiDepthClusterSoAProducer(edm::ParameterSet const& config, PFMultiDepthClusterParamsCache const*);

    void acquire(device::Event const& event, device::EventSetup const&) override;
    void produce(device::Event& event, device::EventSetup const&) override;
    //
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    static std::unique_ptr<PFMultiDepthClusterParamsCache> initializeGlobalCache(edm::ParameterSet const& config);
    static void globalEndJob(PFMultiDepthClusterParamsCache*) {}

  private:
    const device::EDGetToken<reco::PFClusterDeviceCollection> inputPFClusterSoA_Token_;
    const device::EDGetToken<reco::PFRecHitFractionDeviceCollection> inputPFRecHitFractionSoA_Token_;
    const device::EDGetToken<reco::PFRecHitDeviceCollection> inputPFRecHitSoA_Token_;

    const device::EDPutToken<reco::PFClusterDeviceCollection> outputPFClusterSoA_Token_;
    const device::EDPutToken<reco::PFRecHitFractionDeviceCollection> outputPFRHFractionsSoA_Token_;

    // data members used to communicate between acquire() and produce()
    cms::alpakatools::host_buffer<int> pfrhfrac_size_;
    cms::alpakatools::host_buffer<int> pfcl_size_;
  };

  void PFMultiDepthClusterSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("clustersSrc", {});
    desc.add<edm::InputTag>("rhfracSrc", {});
    desc.add<edm::InputTag>("rechitSrc", {});
    desc.add<double>("nSigmaEta", 2.0);
    desc.add<double>("nSigmaPhi", 2.0);
    desc.add<edm::ParameterSetDescription>("energyCorrector", {});
    {
      edm::ParameterSetDescription pset0;

      pset0.add<std::string>("algoName", "PFMultiDepthClusterizer");

      {
        edm::ParameterSetDescription pset1;
        pset1.add<std::string>("algoName", "Basic2DGenericPFlowPositionCalc");
        {
          edm::ParameterSetDescription psd;
          psd.add<std::vector<int>>("depths", {});
          psd.add<std::string>("detector", "");
          psd.add<std::vector<double>>("logWeightDenominator", {});
          pset1.addVPSet("logWeightDenominatorByDetector", psd, {});
        }
        pset1.add<double>("minAllowedNormalization", 1e-09);
        pset1.add<double>("minFractionInCalc", 1e-09);
        pset1.add<int>("posCalcNCrystals", -1);
        pset1.add<edm::ParameterSetDescription>("timeResolutionCalcBarrel", {});
        pset1.add<edm::ParameterSetDescription>("timeResolutionCalcEndcap", {});
        pset0.add<edm::ParameterSetDescription>("allCellsPositionCalc", pset1);
      }

      pset0.add<double>("minFractionToKeep", 1e-07);
      pset0.add<double>("nSigmaEta", 2.0);
      pset0.add<double>("nSigmaPhi", 2.0);
      desc.add<edm::ParameterSetDescription>("pfClusterBuilder", pset0);
    }
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<PFMultiDepthClusterParamsCache> PFMultiDepthClusterSoAProducer::initializeGlobalCache(
      edm::ParameterSet const& config) {
    PortableHostObject<PFMultiDepthClusterParams> params(cms::alpakatools::host());

    params->nSigmaEta = std::pow(config.getParameter<double>("nSigmaEta"), 2.0);
    params->nSigmaPhi = std::pow(config.getParameter<double>("nSigmaPhi"), 2.0);

    return std::make_unique<PFMultiDepthClusterParamsCache>(std::move(params));
  }

  PFMultiDepthClusterSoAProducer::PFMultiDepthClusterSoAProducer(const edm::ParameterSet& config,
                                                                 PFMultiDepthClusterParamsCache const*)
      : SynchronizingEDProducer(config),
        inputPFClusterSoA_Token_{consumes(config.getParameter<edm::InputTag>("clustersSrc"))},
        inputPFRecHitFractionSoA_Token_{consumes(config.getParameter<edm::InputTag>("rhfracSrc"))},
        inputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("rechitSrc"))},
        outputPFClusterSoA_Token_{produces()},
        outputPFRHFractionsSoA_Token_{produces()},
        pfrhfrac_size_{cms::alpakatools::make_host_buffer<int, Platform>()},
        pfcl_size_{cms::alpakatools::make_host_buffer<int, Platform>()} {}

  void PFMultiDepthClusterSoAProducer::acquire(device::Event const& event, device::EventSetup const&) {
    *pfrhfrac_size_ = 0;
    *pfcl_size_ = 0;

    auto const& pfClusters_ = event.get(inputPFClusterSoA_Token_);

    auto pfrhfrac_size_d =
        cms::alpakatools::make_device_view<const int>(event.queue(), pfClusters_.const_view().nRHFracs());
    alpaka::memcpy(event.queue(), pfrhfrac_size_, pfrhfrac_size_d);

    auto pfcl_size_d = cms::alpakatools::make_device_view<const int>(event.queue(), pfClusters_.const_view().nSeeds());
    alpaka::memcpy(event.queue(), pfcl_size_, pfcl_size_d);
  }

  void PFMultiDepthClusterSoAProducer::produce(device::Event& event, const device::EventSetup& eventSetup) {
    const reco::PFClusterDeviceCollection& pfClusters = event.get(inputPFClusterSoA_Token_);
    const reco::PFRecHitDeviceCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);

    const reco::PFRecHitFractionDeviceCollection& pfRecHitFractions = event.get(inputPFRecHitFractionSoA_Token_);

    auto const* paramsDev = globalCache()->get(event.queue()).const_data();

    std::unique_ptr<reco::PFRecHitFractionDeviceCollection> outPFRHFractions;
    std::unique_ptr<reco::PFClusterDeviceCollection> outPFClusters;

    LogDebug("PFMultiDepthClusterSoAProducer") << "nClusters is: " << *pfcl_size_;

    if (*pfcl_size_ > 0) {
      outPFClusters = std::make_unique<reco::PFClusterDeviceCollection>(event.queue(), *pfcl_size_);
      outPFRHFractions = std::make_unique<reco::PFRecHitFractionDeviceCollection>(event.queue(), *pfrhfrac_size_);

      eclcc::clusterize(event.queue(),
                        *outPFClusters,
                        *outPFRHFractions,
                        pfClusters,
                        pfRecHitFractions,
                        pfRecHits,
                        paramsDev,
                        *pfcl_size_);
    } else {
      outPFClusters = std::make_unique<reco::PFClusterDeviceCollection>(event.queue(), 0);
      outPFRHFractions = std::make_unique<reco::PFRecHitFractionDeviceCollection>(event.queue(), 0);
    }

    event.emplace(outputPFClusterSoA_Token_, std::move(*outPFClusters));
    event.emplace(outputPFRHFractionsSoA_Token_, std::move(*outPFRHFractions));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFMultiDepthClusterSoAProducer);

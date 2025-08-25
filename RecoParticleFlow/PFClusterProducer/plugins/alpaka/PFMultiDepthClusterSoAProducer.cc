#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
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
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizer_Alpaka.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    using PFMultiDepthClusterParamsCache =
        cms::alpakatools::MoveToDeviceCache<Device, PortableHostObject<PFMultiDepthClusterParams>>;
  }	

  class PFMultiDepthClusterSoAProducer : public stream::SynchronizingEDProducer<edm::GlobalCache<PFMultiDepthClusterParamsCache>> {
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

    const edm::EDGetTokenT<cms_uint32_t> inputPFClustersNum_Token_;
    const edm::EDGetTokenT<cms_uint32_t> inputPFRecHitNum_Token_;

    const device::EDPutToken<reco::PFClusterDeviceCollection> outputPFClusterSoA_Token_;
    const device::EDPutToken<reco::PFRecHitFractionDeviceCollection> outputPFRHFractionsSoA_Token_;
    
    std::optional<PFMultiDepthClusterizer_Alpaka> clusterizer_;
  };

  void PFMultiDepthClusterSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("clustersSource", {});
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

  std::unique_ptr<PFMultiDepthClusterParamsCache> PFMultiDepthClusterSoAProducer::initializeGlobalCache(edm::ParameterSet const& config) {
    PortableHostObject<PFMultiDepthClusterParams> params(cms::alpakatools::host());
  
    params->nSigmaEta = std::pow(config.getParameter<double>("nSigmaEta"), 2.0);
    params->nSigmaPhi = std::pow(config.getParameter<double>("nSigmaPhi"), 2.0);

    return std::make_unique<PFMultiDepthClusterParamsCache>(std::move(params));
  }

  PFMultiDepthClusterSoAProducer::PFMultiDepthClusterSoAProducer(const edm::ParameterSet& config, PFMultiDepthClusterParamsCache const*)
      : SynchronizingEDProducer(config),
        inputPFClusterSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfClusters"))},
        inputPFRecHitFractionSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfRecHits"))},
        inputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfRecHits"))},
        outputPFClusterSoA_Token_{produces()},
        outputPFRHFractionsSoA_Token_{produces()}{}

  void PFMultiDepthClusterSoAProducer::acquire(device::Event const& event, device::EventSetup const&) {

    if (!clusterizer_) {
      // Initialize clusterizer at first event
      clusterizer_.emplace();
    }
  }

  void PFMultiDepthClusterSoAProducer::produce(device::Event& event, const device::EventSetup& eventSetup) {
    const reco::PFClusterDeviceCollection& pfClusters = event.get(inputPFClusterSoA_Token_);
    const reco::PFRecHitDeviceCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);

    const reco::PFRecHitFractionDeviceCollection& pfRecHitFractions = event.get(inputPFRecHitFractionSoA_Token_);

    auto const* paramsDev = globalCache()->get(event.queue()).const_data();

    std::unique_ptr<reco::PFRecHitFractionDeviceCollection> outPFRHFractions;
    std::unique_ptr<reco::PFClusterDeviceCollection> outPFClusters;

    int nClusters_ = event.get(inputPFClustersNum_Token_);

    if (nClusters_ > 0) {
      int nRH_ = event.get(inputPFRecHitNum_Token_);
      
      outPFClusters = std::make_unique<reco::PFClusterDeviceCollection>(nClusters_, event.queue());
      outPFRHFractions = std::make_unique<reco::PFRecHitFractionDeviceCollection>(nRH_, event.queue());
      
      clusterizer_->apply(event.queue(), *outPFClusters, *outPFRHFractions, pfClusters, pfRecHitFractions, pfRecHits, paramsDev, nClusters_);
    } else {
      outPFClusters = std::make_unique<reco::PFClusterDeviceCollection>(0, event.queue());
      outPFRHFractions = std::make_unique<reco::PFRecHitFractionDeviceCollection>(0, event.queue());
    }

    event.emplace(outputPFClusterSoA_Token_, std::move(*outPFClusters));
    event.emplace(outputPFRHFractionsSoA_Token_, std::move(*outPFRHFractions));
  }

  #include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
  DEFINE_FWK_ALPAKA_MODULE(PFMultiDepthClusterSoAProducer);


}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

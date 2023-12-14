#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFClusterParamsDeviceCollection.h"

#include <array>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFClusterParamsESProducer : public ESProducer {
  public:
    PFClusterParamsESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      constexpr static uint32_t kMaxDepth_barrel = 4;
      constexpr static uint32_t kMaxDepth_endcap = 7;
      product = std::make_shared<reco::PFClusterParamsHostCollection>(std::max(kMaxDepth_barrel, kMaxDepth_endcap),
                                                                      cms::alpakatools::host());
      auto view = product->view();

      // seedFinder
      auto const& sfConf = iConfig.getParameterSet("seedFinder");
      view.nNeigh() = sfConf.getParameter<int>("nNeighbours");
      auto const& seedFinderConfs = sfConf.getParameterSetVector("thresholdsByDetector");
      for (auto const& pset : seedFinderConfs) {
        auto const& det = pset.getParameter<std::string>("detector");
        auto seedPt2Threshold = std::pow(pset.getParameter<double>("seedingThresholdPt"), 2.);
        auto const& thresholds = pset.getParameter<std::vector<double>>("seedingThreshold");
        if (det == "HCAL_BARREL1") {
          if (thresholds.size() != kMaxDepth_barrel)
            throw cms::Exception("Configuration") << "Invalid size (" << thresholds.size() << " != " << kMaxDepth_barrel
                                                  << ") for \"\" vector of det = \"" << det << "\"";
          view.seedPt2ThresholdHB() = seedPt2Threshold;
          for (size_t idx = 0; idx < thresholds.size(); ++idx) {
            view.seedEThresholdHB_vec()[idx] = thresholds[idx];
          }
        } else if (det == "HCAL_ENDCAP") {
          if (thresholds.size() != kMaxDepth_endcap)
            throw cms::Exception("Configuration") << "Invalid size (" << thresholds.size() << " != " << kMaxDepth_endcap
                                                  << ") for \"\" vector of det = \"" << det << "\"";
          view.seedPt2ThresholdHE() = seedPt2Threshold;
          for (size_t idx = 0; idx < thresholds.size(); ++idx) {
            view.seedEThresholdHE_vec()[idx] = thresholds[idx];
          }
        } else {
          throw cms::Exception("Configuration") << "Unknown detector when parsing seedFinder: " << det;
        }
      }

      // initialClusteringStep
      auto const& initConf = iConfig.getParameterSet("initialClusteringStep");
      auto const& topoThresholdConf = initConf.getParameterSetVector("thresholdsByDetector");
      for (auto const& pset : topoThresholdConf) {
        auto const& det = pset.getParameter<std::string>("detector");
        auto const& thresholds = pset.getParameter<std::vector<double>>("gatheringThreshold");
        if (det == "HCAL_BARREL1") {
          if (thresholds.size() != kMaxDepth_barrel)
            throw cms::Exception("Configuration") << "Invalid size (" << thresholds.size() << " != " << kMaxDepth_barrel
                                                  << ") for \"\" vector of det = \"" << det << "\"";
          for (size_t idx = 0; idx < thresholds.size(); ++idx) {
            view.topoEThresholdHB_vec()[idx] = thresholds[idx];
          }
        } else if (det == "HCAL_ENDCAP") {
          if (thresholds.size() != kMaxDepth_endcap)
            throw cms::Exception("Configuration") << "Invalid size (" << thresholds.size() << " != " << kMaxDepth_endcap
                                                  << ") for \"\" vector of det = \"" << det << "\"";
          for (size_t idx = 0; idx < thresholds.size(); ++idx) {
            view.topoEThresholdHE_vec()[idx] = thresholds[idx];
          }
        } else {
          throw cms::Exception("Configuration") << "Unknown detector when parsing initClusteringStep: " << det;
        }
      }

      // pfClusterBuilder
      auto const& pfClusterPSet = iConfig.getParameterSet("pfClusterBuilder");
      view.showerSigma2() = std::pow(pfClusterPSet.getParameter<double>("showerSigma"), 2.);
      view.minFracToKeep() = pfClusterPSet.getParameter<double>("minFractionToKeep");
      view.minFracTot() = pfClusterPSet.getParameter<double>("minFracTot");
      view.maxIterations() = pfClusterPSet.getParameter<unsigned int>("maxIterations");
      view.excludeOtherSeeds() = pfClusterPSet.getParameter<bool>("excludeOtherSeeds");
      view.stoppingTolerance() = pfClusterPSet.getParameter<double>("stoppingTolerance");
      auto const& pcPSet = pfClusterPSet.getParameterSet("positionCalc");
      view.minFracInCalc() = pcPSet.getParameter<double>("minFractionInCalc");
      view.minAllowedNormalization() = pcPSet.getParameter<double>("minAllowedNormalization");

      auto const& recHitEnergyNormConf = pfClusterPSet.getParameterSetVector("recHitEnergyNorms");
      for (auto const& pset : recHitEnergyNormConf) {
        auto const& recHitNorms = pset.getParameter<std::vector<double>>("recHitEnergyNorm");
        auto const& det = pset.getParameter<std::string>("detector");
        if (det == "HCAL_BARREL1") {
          if (recHitNorms.size() != kMaxDepth_barrel)
            throw cms::Exception("Configuration")
                << "Invalid size (" << recHitNorms.size() << " != " << kMaxDepth_barrel
                << ") for \"\" vector of det = \"" << det << "\"";
          for (size_t idx = 0; idx < recHitNorms.size(); ++idx) {
            view.recHitEnergyNormInvHB_vec()[idx] = 1. / recHitNorms[idx];
          }
        } else if (det == "HCAL_ENDCAP") {
          if (recHitNorms.size() != kMaxDepth_endcap)
            throw cms::Exception("Configuration")
                << "Invalid size (" << recHitNorms.size() << " != " << kMaxDepth_endcap
                << ") for \"\" vector of det = \"" << det << "\"";
          for (size_t idx = 0; idx < recHitNorms.size(); ++idx) {
            view.recHitEnergyNormInvHE_vec()[idx] = 1. / recHitNorms[idx];
          }
        } else {
          throw cms::Exception("Configuration") << "Unknown detector when parsing recHitEnergyNorms: " << det;
        }
      }

      auto const& barrelTimeResConf = pfClusterPSet.getParameterSet("timeResolutionCalcBarrel");
      view.barrelTimeResConsts_corrTermLowE() = barrelTimeResConf.getParameter<double>("corrTermLowE");
      view.barrelTimeResConsts_threshLowE() = barrelTimeResConf.getParameter<double>("threshLowE");
      view.barrelTimeResConsts_noiseTerm() = barrelTimeResConf.getParameter<double>("noiseTerm");
      view.barrelTimeResConsts_constantTermLowE2() =
          std::pow(barrelTimeResConf.getParameter<double>("constantTermLowE"), 2.);
      view.barrelTimeResConsts_noiseTermLowE() = barrelTimeResConf.getParameter<double>("noiseTermLowE");
      view.barrelTimeResConsts_threshHighE() = barrelTimeResConf.getParameter<double>("threshHighE");
      view.barrelTimeResConsts_constantTerm2() = std::pow(barrelTimeResConf.getParameter<double>("constantTerm"), 2.);
      view.barrelTimeResConsts_resHighE2() =
          std::pow(view.barrelTimeResConsts_noiseTerm() / view.barrelTimeResConsts_threshHighE(), 2.) +
          view.barrelTimeResConsts_constantTerm2();

      auto const& endcapTimeResConf = pfClusterPSet.getParameterSet("timeResolutionCalcEndcap");
      view.endcapTimeResConsts_corrTermLowE() = endcapTimeResConf.getParameter<double>("corrTermLowE");
      view.endcapTimeResConsts_threshLowE() = endcapTimeResConf.getParameter<double>("threshLowE");
      view.endcapTimeResConsts_noiseTerm() = endcapTimeResConf.getParameter<double>("noiseTerm");
      view.endcapTimeResConsts_constantTermLowE2() =
          std::pow(endcapTimeResConf.getParameter<double>("constantTermLowE"), 2.);
      view.endcapTimeResConsts_noiseTermLowE() = endcapTimeResConf.getParameter<double>("noiseTermLowE");
      view.endcapTimeResConsts_threshHighE() = endcapTimeResConf.getParameter<double>("threshHighE");
      view.endcapTimeResConsts_constantTerm2() = std::pow(endcapTimeResConf.getParameter<double>("constantTerm"), 2.);
      view.endcapTimeResConsts_resHighE2() =
          std::pow(view.endcapTimeResConsts_noiseTerm() / view.endcapTimeResConsts_threshHighE(), 2.) +
          view.endcapTimeResConsts_constantTerm2();

      setWhatProduced(this);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription psetDesc;
      {
        auto const psetName = "seedFinder";
        edm::ParameterSetDescription foo;
        foo.add<int>("nNeighbours", 4);
        {
          edm::ParameterSetDescription validator;
          validator.add<std::string>("detector", "");
          validator.add<std::vector<double>>("seedingThreshold", {});
          validator.add<double>("seedingThresholdPt", 0.);
          std::vector<edm::ParameterSet> vDefaults(2);
          vDefaults[0].addParameter<std::string>("detector", "HCAL_BARREL1");
          vDefaults[0].addParameter<std::vector<double>>("seedingThreshold", {0.125, 0.25, 0.35, 0.35});
          vDefaults[0].addParameter<double>("seedingThresholdPt", 0.);
          vDefaults[1].addParameter<std::string>("detector", "HCAL_ENDCAP");
          vDefaults[1].addParameter<std::vector<double>>("seedingThreshold",
                                                         {0.1375, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275});
          vDefaults[1].addParameter<double>("seedingThresholdPt", 0.);
          foo.addVPSet("thresholdsByDetector", validator, vDefaults);
        }
        psetDesc.add(psetName, foo);
      }
      {
        auto const psetName = "initialClusteringStep";
        edm::ParameterSetDescription foo;
        {
          edm::ParameterSetDescription validator;
          validator.add<std::string>("detector", "");
          validator.add<std::vector<double>>("gatheringThreshold", {});
          std::vector<edm::ParameterSet> vDefaults(2);
          vDefaults[0].addParameter<std::string>("detector", "HCAL_BARREL1");
          vDefaults[0].addParameter<std::vector<double>>("gatheringThreshold", {0.1, 0.2, 0.3, 0.3});
          vDefaults[1].addParameter<std::string>("detector", "HCAL_ENDCAP");
          vDefaults[1].addParameter<std::vector<double>>("gatheringThreshold", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
          foo.addVPSet("thresholdsByDetector", validator, vDefaults);
        }
        psetDesc.add(psetName, foo);
      }
      {
        auto const psetName = "pfClusterBuilder";
        edm::ParameterSetDescription foo;
        foo.add<unsigned int>("maxIterations", 50);
        foo.add<double>("minFracTot", 1e-20);
        foo.add<double>("minFractionToKeep", 1e-7);
        foo.add<bool>("excludeOtherSeeds", true);
        foo.add<double>("showerSigma", 10.);
        foo.add<double>("stoppingTolerance", 1e-8);
        {
          edm::ParameterSetDescription validator;
          validator.add<std::string>("detector", "");
          validator.add<std::vector<double>>("recHitEnergyNorm", {});
          std::vector<edm::ParameterSet> vDefaults(2);
          vDefaults[0].addParameter<std::string>("detector", "HCAL_BARREL1");
          vDefaults[0].addParameter<std::vector<double>>("recHitEnergyNorm", {0.1, 0.2, 0.3, 0.3});
          vDefaults[1].addParameter<std::string>("detector", "HCAL_ENDCAP");
          vDefaults[1].addParameter<std::vector<double>>("recHitEnergyNorm", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
          foo.addVPSet("recHitEnergyNorms", validator, vDefaults);
        }
        {
          edm::ParameterSetDescription bar;
          bar.add<double>("minFractionInCalc", 1e-9);
          bar.add<double>("minAllowedNormalization", 1e-9);
          foo.add("positionCalc", bar);
        }
        {
          edm::ParameterSetDescription bar;
          bar.add<double>("corrTermLowE", 0.);
          bar.add<double>("threshLowE", 6.);
          bar.add<double>("noiseTerm", 21.86);
          bar.add<double>("constantTermLowE", 4.24);
          bar.add<double>("noiseTermLowE", 8.);
          bar.add<double>("threshHighE", 15.);
          bar.add<double>("constantTerm", 2.82);
          foo.add("timeResolutionCalcBarrel", bar);
        }
        {
          edm::ParameterSetDescription bar;
          bar.add<double>("corrTermLowE", 0.);
          bar.add<double>("threshLowE", 6.);
          bar.add<double>("noiseTerm", 21.86);
          bar.add<double>("constantTermLowE", 4.24);
          bar.add<double>("noiseTermLowE", 8.);
          bar.add<double>("threshHighE", 15.);
          bar.add<double>("constantTerm", 2.82);
          foo.add("timeResolutionCalcEndcap", bar);
        }
        psetDesc.add(psetName, foo);
      }

      descriptions.addWithDefaultLabel(psetDesc);
    }

    std::shared_ptr<reco::PFClusterParamsHostCollection> produce(JobConfigurationGPURecord const& iRecord) {
      return product;
    }

  private:
    std::shared_ptr<reco::PFClusterParamsHostCollection> product;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFClusterParamsESProducer);

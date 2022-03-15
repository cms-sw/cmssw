#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"

std::unique_ptr<StripClusterizerAlgorithm> StripClusterizerAlgorithmFactory::create(edm::ConsumesCollector&& iC,
                                                                                    const edm::ParameterSet& conf) {
  std::string algorithm = conf.getParameter<std::string>("Algorithm");

  if (algorithm == "ThreeThresholdAlgorithm") {
    return std::unique_ptr<StripClusterizerAlgorithm>(new ThreeThresholdAlgorithm(
        iC.esConsumes<SiStripClusterizerConditions, SiStripClusterizerConditionsRcd>(
            edm::ESInputTag{"", conf.getParameter<std::string>("ConditionsLabel")}),
        conf.getParameter<double>("ChannelThreshold"),
        conf.getParameter<double>("SeedThreshold"),
        conf.getParameter<double>("ClusterThreshold"),
        conf.getParameter<unsigned>("MaxSequentialHoles"),
        conf.getParameter<unsigned>("MaxSequentialBad"),
        conf.getParameter<unsigned>("MaxAdjacentBad"),
        // existsAs test should be removed once MaxClusterSize is in the HLT config
        conf.existsAs<unsigned>("MaxClusterSize") ? conf.getParameter<unsigned>("MaxClusterSize") : 3U * 256U,
        conf.getParameter<bool>("RemoveApvShots"),
        clusterChargeCut(conf)));
  }

  if (algorithm == "OldThreeThresholdAlgorithm") {
    throw cms::Exception("[StripClusterizerAlgorithmFactory] obsolete") << algorithm << " Obsolete since 7_3_0";
  }

  throw cms::Exception("[StripClusterizerAlgorithmFactory] Unregistered Algorithm")
      << algorithm << " is not a registered StripClusterizerAlgorithm";
}

void StripClusterizerAlgorithmFactory::fillDescriptions(edm::ParameterSetDescription& clusterizer) {
  clusterizer.add<std::string>("Algorithm", "ThreeThresholdAlgorithm");
  clusterizer.add<std::string>("ConditionsLabel", "");
  clusterizer.add("ChannelThreshold", 2.0);
  clusterizer.add("SeedThreshold", 3.0);
  clusterizer.add("ClusterThreshold", 5.0);
  clusterizer.add("MaxSequentialHoles", 0U);
  clusterizer.add("MaxSequentialBad", 1U);
  clusterizer.add("MaxAdjacentBad", 0U);
  clusterizer.addOptional("MaxClusterSize", 3U * 256U);  // eventually should be add()
  clusterizer.add("RemoveApvShots", true);
  clusterizer.add("setDetId", true);
  clusterizer.add("clusterChargeCut", getConfigurationDescription4CCC(CCC::kNone));
}

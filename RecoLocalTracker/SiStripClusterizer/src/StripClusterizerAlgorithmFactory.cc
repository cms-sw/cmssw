#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"

std::unique_ptr<StripClusterizerAlgorithm> StripClusterizerAlgorithmFactory::
create(const edm::ParameterSet& conf) {
  std::string algorithm = conf.getParameter<std::string>("Algorithm");

  if(algorithm == "ThreeThresholdAlgorithm") {
    return std::unique_ptr<StripClusterizerAlgorithm>(
	   new ThreeThresholdAlgorithm(
	       conf.getParameter<double>("ChannelThreshold"),
	       conf.getParameter<double>("SeedThreshold"),
	       conf.getParameter<double>("ClusterThreshold"),
	       conf.getParameter<unsigned>("MaxSequentialHoles"),
	       conf.getParameter<unsigned>("MaxSequentialBad"),
	       conf.getParameter<unsigned>("MaxAdjacentBad"),
	       conf.getParameter<std::string>("QualityLabel"),
	       conf.getParameter<bool>("RemoveApvShots"),
               clusterChargeCut(conf)
           ));
  }

  if(algorithm == "OldThreeThresholdAlgorithm") {
    throw cms::Exception("[StripClusterizerAlgorithmFactory] obsolete")
          << algorithm << " Obsolete since 7_3_0";

  }

  throw cms::Exception("[StripClusterizerAlgorithmFactory] Unregistered Algorithm")
    << algorithm << " is not a registered StripClusterizerAlgorithm";
}

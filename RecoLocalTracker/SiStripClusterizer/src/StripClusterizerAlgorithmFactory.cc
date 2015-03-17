#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/OldThreeThresholdAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"

std::auto_ptr<StripClusterizerAlgorithm> StripClusterizerAlgorithmFactory::
create(const edm::ParameterSet& conf) {
  std::string algorithm = conf.getParameter<std::string>("Algorithm");

  bool setDetId=false;
  if (conf.exists("setDetId"))
    setDetId = conf.getParameter<bool>("setDetId");
  if(algorithm == "ThreeThresholdAlgorithm") {
    return std::auto_ptr<StripClusterizerAlgorithm>(
	   new ThreeThresholdAlgorithm(
	       conf.getParameter<double>("ChannelThreshold"),
	       conf.getParameter<double>("SeedThreshold"),
	       conf.getParameter<double>("ClusterThreshold"),
	       conf.getParameter<unsigned>("MaxSequentialHoles"),
	       conf.getParameter<unsigned>("MaxSequentialBad"),
	       conf.getParameter<unsigned>("MaxAdjacentBad"),
	       conf.getParameter<std::string>("QualityLabel"),
	       setDetId,
	       conf.getParameter<bool>("RemoveApvShots"),
               clusterChargeCut(conf)
           ));
  }

  if(algorithm == "OldThreeThresholdAlgorithm") {
    return std::auto_ptr<StripClusterizerAlgorithm>(
	   new OldThreeThresholdAlgorithm(
	       conf.getParameter<double>("ChannelThreshold"),
	       conf.getParameter<double>("SeedThreshold"),
	       conf.getParameter<double>("ClusterThreshold"),
	       conf.getParameter<unsigned>("MaxSequentialHoles"),
	       conf.getParameter<std::string>("QualityLabel"),
	       setDetId));
  }

  throw cms::Exception("[StripClusterizerAlgorithmFactory] Unregistered Algorithm")
    << algorithm << " is not a registered StripClusterizerAlgorithm";
}

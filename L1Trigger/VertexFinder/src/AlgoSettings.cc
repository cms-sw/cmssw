
#include "L1Trigger/VertexFinder/interface/AlgoSettings.h"


#include "FWCore/Utilities/interface/Exception.h"



namespace l1tVertexFinder {

///=== Get configuration parameters

AlgoSettings::AlgoSettings(const edm::ParameterSet& iConfig) :
  vertex_(iConfig.getParameter<edm::ParameterSet>("VertexReconstruction")),
  vx_distance_(vertex_.getParameter<double>("VertexDistance")),
  vx_resolution_(vertex_.getParameter<double>("VertexResolution")),
  vx_distanceType_(vertex_.getParameter<unsigned int>("DistanceType")),
  vx_minTracks_(vertex_.getParameter<unsigned int>("MinTracks")),
  vx_weightedmean_(vertex_.getParameter<bool>("WeightedMean")),
  vx_chi2cut_(vertex_.getParameter<double>("AVR_chi2cut")),
  tdr_vx_width_(vertex_.getParameter<double>("TP_VertexWidth")),
  vx_TrackMinPt_(vertex_.getParameter<double>("VxMinTrackPt")),
  vx_dbscan_pt_(vertex_.getParameter<double>("DBSCANPtThreshold")),
  vx_dbscan_mintracks_(vertex_.getParameter<unsigned int>("DBSCANMinDensityTracks")),
  vx_kmeans_iterations_(vertex_.getParameter<unsigned int>("KmeansIterations")),
  vx_kmeans_nclusters_(vertex_.getParameter<unsigned int>("KmeansNumClusters")),
  // Debug printout
  debug_(iConfig.getParameter<unsigned int>("Debug"))
{
  const std::string algoName(vertex_.getParameter<std::string>("Algorithm"));
  const auto algoMapIt = algoNameMap.find(algoName);
  if (algoMapIt != algoNameMap.end())
    vx_algo_ = algoMapIt->second;
  else {
    std::ostringstream validAlgoNames;
    for (auto it = algoNameMap.begin(); it != algoNameMap.end(); it++) {
      validAlgoNames << '"' << it->first << '"';
      if (it != (--algoNameMap.end()))
        validAlgoNames << ", ";
    }
    throw cms::Exception("Invalid algo name '" + algoName + "' specified for L1T vertex producer. Valid algo names are: " + validAlgoNames.str());
  }
}


const std::map<std::string, Algorithm> AlgoSettings::algoNameMap = { { "GapClustering", Algorithm::GapClustering },
                                                                     { "Agglomerative", Algorithm::AgglomerativeHierarchical },
                                                                     { "DBSCAN", Algorithm::DBSCAN },
                                                                     { "PVR", Algorithm::PVR },
                                                                     { "Adaptive", Algorithm::AdaptiveVertexReconstruction },
                                                                     { "HPV", Algorithm::HPV },
                                                                     { "K-means", Algorithm::Kmeans } };


} // end namespace l1tVertexFinder

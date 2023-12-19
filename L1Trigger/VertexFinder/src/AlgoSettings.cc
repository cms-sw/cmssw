#include "L1Trigger/VertexFinder/interface/AlgoSettings.h"

namespace l1tVertexFinder {

  ///=== Get configuration parameters

  AlgoSettings::AlgoSettings(const edm::ParameterSet& iConfig)
      : vertex_(iConfig.getParameter<edm::ParameterSet>("VertexReconstruction")),
        vx_distance_(vertex_.getParameter<double>("VertexDistance")),
        vx_resolution_(vertex_.getParameter<double>("VertexResolution")),
        vx_distanceType_(vertex_.getParameter<unsigned int>("DistanceType")),
        vx_minTracks_(vertex_.getParameter<unsigned int>("MinTracks")),
        vx_weightedmean_(vertex_.getParameter<unsigned int>("WeightedMean")),
        vx_chi2cut_(vertex_.getParameter<double>("AVR_chi2cut")),
        vx_DoQualityCuts_(vertex_.getParameter<bool>("EM_DoQualityCuts")),
        vx_DoPtComp_(vertex_.getParameter<bool>("FH_DoPtComp")),
        vx_DoTightChi2_(vertex_.getParameter<bool>("FH_DoTightChi2")),
        vx_histogram_parameters_(vertex_.getParameter<std::vector<double> >("FH_HistogramParameters")),
        vx_nvtx_(vertex_.getParameter<unsigned int>("FH_NVtx")),
        vx_width_(vertex_.getParameter<double>("FH_VertexWidth")),
        vx_windowSize_(vertex_.getParameter<unsigned int>("FH_WindowSize")),
        vx_TrackMinPt_(vertex_.getParameter<double>("VxMinTrackPt")),
        vx_TrackMaxPt_(vertex_.getParameter<double>("VxMaxTrackPt")),
        vx_TrackMaxPtBehavior_(vertex_.getParameter<int>("VxMaxTrackPtBehavior")),
        vx_TrackMaxChi2_(vertex_.getParameter<double>("VxMaxTrackChi2")),
        vx_NStubMin_(vertex_.getParameter<unsigned int>("VxMinNStub")),
        vx_NStubPSMin_(vertex_.getParameter<unsigned int>("VxMinNStubPS")),
        vx_dbscan_pt_(vertex_.getParameter<double>("DBSCANPtThreshold")),
        vx_dbscan_mintracks_(vertex_.getParameter<unsigned int>("DBSCANMinDensityTracks")),
        vx_kmeans_iterations_(vertex_.getParameter<unsigned int>("KmeansIterations")),
        vx_kmeans_nclusters_(vertex_.getParameter<unsigned int>("KmeansNumClusters")),
        vx_trkw_graph_(vertex_.getParameter<std::string>("TrackWeightGraph")),
        vx_pvz0_graph_(vertex_.getParameter<std::string>("PVZ0Graph")),
        // Debug printout
        debug_(iConfig.getParameter<unsigned int>("debug")) {
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
      throw cms::Exception("Invalid algo name '" + algoName +
                           "' specified for L1T vertex producer. Valid algo names are: " + validAlgoNames.str());
    }

    const auto algoPrecisionMapIt = algoPrecisionMap.find(vx_algo_);
    if (algoPrecisionMapIt != algoPrecisionMap.end()) {
      vx_precision_ = algoPrecisionMapIt->second;
    } else {
      throw cms::Exception("Unknown precision {Simulation, Emulation} for algo name " + algoName);
    }
  }

  const std::map<std::string, Algorithm> AlgoSettings::algoNameMap = {
      {"fastHisto", Algorithm::fastHisto},
      {"fastHistoEmulation", Algorithm::fastHistoEmulation},
      {"fastHistoLooseAssociation", Algorithm::fastHistoLooseAssociation},
      {"GapClustering", Algorithm::GapClustering},
      {"agglomerative", Algorithm::agglomerativeHierarchical},
      {"DBSCAN", Algorithm::DBSCAN},
      {"PVR", Algorithm::PVR},
      {"adaptive", Algorithm::adaptiveVertexReconstruction},
      {"HPV", Algorithm::HPV},
      {"K-means", Algorithm::Kmeans},
      {"NNEmulation", Algorithm::NNEmulation}};

  const std::map<Algorithm, Precision> AlgoSettings::algoPrecisionMap = {
      {Algorithm::fastHisto, Precision::Simulation},
      {Algorithm::fastHistoEmulation, Precision::Emulation},
      {Algorithm::fastHistoLooseAssociation, Precision::Simulation},
      {Algorithm::GapClustering, Precision::Simulation},
      {Algorithm::agglomerativeHierarchical, Precision::Simulation},
      {Algorithm::DBSCAN, Precision::Simulation},
      {Algorithm::PVR, Precision::Simulation},
      {Algorithm::adaptiveVertexReconstruction, Precision::Simulation},
      {Algorithm::HPV, Precision::Simulation},
      {Algorithm::Kmeans, Precision::Simulation},
      {Algorithm::NNEmulation, Precision::Emulation}};

}  // end namespace l1tVertexFinder

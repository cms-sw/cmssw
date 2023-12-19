#ifndef __L1Trigger_VertexFinder_AlgoSettings_h__
#define __L1Trigger_VertexFinder_AlgoSettings_h__

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace l1tVertexFinder {

  enum class Algorithm {
    fastHisto,
    fastHistoEmulation,
    fastHistoLooseAssociation,
    GapClustering,
    agglomerativeHierarchical,
    DBSCAN,
    PVR,
    adaptiveVertexReconstruction,
    HPV,
    Kmeans,
    NNEmulation
  };

  enum class Precision { Simulation, Emulation };

  class AlgoSettings {
  public:
    AlgoSettings(const edm::ParameterSet& iConfig);
    ~AlgoSettings() {}

    //=== Vertex Reconstruction configuration
    // Vertex Reconstruction algo
    Algorithm vx_algo() const { return vx_algo_; }
    Precision vx_precision() const { return vx_precision_; }
    // For agglomerative cluster algorithm, select a definition of distance between clusters
    unsigned int vx_distanceType() const { return vx_distanceType_; }
    // Assumed Vertex Distance
    float vx_distance() const { return vx_distance_; }
    // Assumed Vertex Resolution
    float vx_resolution() const { return vx_resolution_; }
    // Minimum number of tracks to accept vertex
    unsigned int vx_minTracks() const { return vx_minTracks_; }
    // Compute the z0 position of the vertex with a mean weighted with track momenta
    unsigned int vx_weightedmean() const { return vx_weightedmean_; }
    // Chi2 cut for the adaptive Vertex Recostruction Algorithm
    float vx_chi2cut() const { return vx_chi2cut_; }
    // Do track quality cuts in emulation algorithms
    bool vx_DoQualityCuts() const { return vx_DoQualityCuts_; }
    // Window size of the sliding window
    unsigned int vx_windowSize() const { return vx_windowSize_; }
    // fastHisto histogram parameters (min, max, width)
    std::vector<double> vx_histogram_parameters() const { return vx_histogram_parameters_; }
    double vx_histogram_min() const { return vx_histogram_parameters_.at(0); }
    double vx_histogram_max() const { return vx_histogram_parameters_.at(1); }
    double vx_histogram_binwidth() const { return vx_histogram_parameters_.at(2); }
    int vx_histogram_numbins() const {
      return (vx_histogram_parameters_.at(1) - vx_histogram_parameters_.at(0)) / vx_histogram_parameters_.at(2);
    }
    // fastHisto assumed vertex width
    float vx_width() const { return vx_width_; }
    // fastHisto track selection control
    bool vx_DoPtComp() const { return vx_DoPtComp_; }
    bool vx_DoTightChi2() const { return vx_DoTightChi2_; }
    // Number of vertices to return for fastHisto
    unsigned int vx_nvtx() const { return vx_nvtx_; }
    float vx_dbscan_pt() const { return vx_dbscan_pt_; }
    unsigned int vx_dbscan_mintracks() const { return vx_dbscan_mintracks_; }

    unsigned int vx_kmeans_iterations() const { return vx_kmeans_iterations_; }
    unsigned int vx_kmeans_nclusters() const { return vx_kmeans_nclusters_; }
    float vx_TrackMinPt() const { return vx_TrackMinPt_; }
    float vx_TrackMaxPt() const { return vx_TrackMaxPt_; }
    float vx_TrackMaxPtBehavior() const { return vx_TrackMaxPtBehavior_; }
    float vx_TrackMaxChi2() const { return vx_TrackMaxChi2_; }
    unsigned int vx_NStubMin() const { return vx_NStubMin_; }
    unsigned int vx_NStubPSMin() const { return vx_NStubPSMin_; }

    // Functions for NN:
    std::string vx_trkw_graph() const { return vx_trkw_graph_; }
    std::string vx_pvz0_graph() const { return vx_pvz0_graph_; }

    //=== Debug printout
    unsigned int debug() const { return debug_; }

    //=== Hard-wired constants
    // EJC Check this.  Found stub at r = 109.504 with flat geometry in 81X, so increased tracker radius for now.
    double trackerOuterRadius() const { return 120.2; }  // max. occuring stub radius.
    // EJC Check this.  Found stub at r = 20.664 with flat geometry in 81X, so decreased tracker radius for now.
    double trackerInnerRadius() const { return 20; }   // min. occuring stub radius.
    double trackerHalfLength() const { return 270.; }  // half-length  of tracker.
    double layerIDfromRadiusBin() const {
      return 6.;
    }  // When counting stubs in layers, actually histogram stubs in distance from beam-line with this bin size.

  private:
    static const std::map<std::string, Algorithm> algoNameMap;
    static const std::map<Algorithm, Precision> algoPrecisionMap;

    // Parameter sets for differents types of configuration parameter.
    edm::ParameterSet vertex_;

    // Vertex Reconstruction configuration
    Algorithm vx_algo_;
    Precision vx_precision_;
    float vx_distance_;
    float vx_resolution_;
    unsigned int vx_distanceType_;
    unsigned int vx_minTracks_;
    unsigned int vx_weightedmean_;
    float vx_chi2cut_;
    bool vx_DoQualityCuts_;
    bool vx_DoPtComp_;
    bool vx_DoTightChi2_;
    std::vector<double> vx_histogram_parameters_;
    unsigned int vx_nvtx_;
    float vx_width_;
    unsigned int vx_windowSize_;
    float vx_TrackMinPt_;
    float vx_TrackMaxPt_;
    int vx_TrackMaxPtBehavior_;
    float vx_TrackMaxChi2_;
    unsigned int vx_NStubMin_;
    unsigned int vx_NStubPSMin_;
    float vx_dbscan_pt_;
    float vx_dbscan_mintracks_;
    unsigned int vx_kmeans_iterations_;
    unsigned int vx_kmeans_nclusters_;
    std::string vx_trkw_graph_;
    std::string vx_pvz0_graph_;
    // Debug printout
    unsigned int debug_;
  };

}  // end namespace l1tVertexFinder

#endif

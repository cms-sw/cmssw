#ifndef __L1Trigger_VertexFinder_Settings_h__
#define __L1Trigger_VertexFinder_Settings_h__
 

#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
 
 

namespace l1tVertexFinder {

enum class Algorithm {
  GapClustering,
  AgglomerativeHierarchical,
  DBSCAN,
  PVR,
  AdaptiveVertexReconstruction,
  HPV,
  Kmeans
};

// Stores all configuration parameters + some hard-wired constants.
 
class Settings {
 
public:
  Settings(const edm::ParameterSet& iConfig);
  ~Settings(){}

  //=== Cuts on MC truth tracks for tracking efficiency measurements.
 
  double               genMinPt()                const   {return genMinPt_;}
  double               genMaxAbsEta()            const   {return genMaxAbsEta_;}
  double               genMaxVertR()             const   {return genMaxVertR_;}
  double               genMaxVertZ()             const   {return genMaxVertZ_;}
  const std::vector<int>&   genPdgIds()               const   {return genPdgIds_;}
  // Additional cut on MC truth tracks for algorithmic tracking efficiency measurements.
  unsigned int         genMinStubLayers()        const   {return genMinStubLayers_;} // Min. number of layers TP made stub in.

  //=== Rules for deciding when the track finding has found an L1 track candidate
 
  // Define layers using layer ID (true) or by bins in radius of 5 cm width (false)?
  bool                 useLayerID()              const   {return useLayerID_;}
  //Reduce this layer ID, so that it takes no more than 8 different values in any eta region (simplifies firmware)?
  bool                 reduceLayerID()           const   {return reduceLayerID_;}
 
  //=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
 
  //--- Three different ways to define if a tracking particle matches a reco track candidate. (Usually, set two of them to ultra loose).
  // Min. fraction of matched stubs relative to number of stubs on reco track.
  double               minFracMatchStubsOnReco() const   {return minFracMatchStubsOnReco_;}
  // Min. fraction of matched stubs relative to number of stubs on tracking particle.
  double               minFracMatchStubsOnTP()   const   {return minFracMatchStubsOnTP_;}
  // Min. number of matched layers & min. number of matched PS layers..
  unsigned int         minNumMatchLayers()       const   {return minNumMatchLayers_;}
  unsigned int         minNumMatchPSLayers()     const   {return minNumMatchPSLayers_;}
  // Associate stub to TP only if the TP contributed to both its clusters? (If False, then associate even if only one cluster was made by TP).
  bool                 stubMatchStrict()         const   {return stubMatchStrict_;}

   //=== Vertex Reconstruction configuration
  // Vertex Reconstruction algo
  Algorithm           vx_algo()                  const {return vx_algo_;        }
  /// For Agglomerative cluster algorithm, select a definition of distance between clusters
  unsigned int        vx_distanceType()          const {return vx_distanceType_;  }
  /// Keep only PV from hard interaction (highest pT)
  bool                vx_keepOnlyPV()            const {return vx_keepOnlyPV_;    }
  // Assumed Vertex Distance
  float               vx_distance()              const {return vx_distance_;      }
  // Assumed Vertex Resolution
  float               vx_resolution()            const {return vx_resolution_;    }
  // Minimum number of tracks to accept vertex
  unsigned int        vx_minTracks()             const {return vx_minTracks_;     }
  // Compute the z0 position of the vertex with a mean weighted with track momenta
  bool                vx_weightedmean()          const {return vx_weightedmean_;     }
  /// Chi2 cut for the Adaptive Vertex Recostruction Algorithm
  float               vx_chi2cut()               const {return vx_chi2cut_;       }
  /// TDR assumed vertex width
  float               tdr_vx_width()             const {return tdr_vx_width_;     }
  /// Run the Vertex reconstruction locally or globally
  bool                vx_local()                 const {return vx_local_;         }
  /// Maximum distance to merge vertices
  float               vx_merge_distance()        const {return vx_merge_distance_;}
  float               vx_dbscan_pt()             const {return vx_dbscan_pt_;}
  unsigned int        vx_dbscan_mintracks()      const {return vx_dbscan_mintracks_;}


  /// If running the vertex reconstruction locally, do it individually per HT sector (if false do it by octants)
  unsigned int        vx_kmeans_iterations()     const {return vx_kmeans_iterations_;}
  unsigned int        vx_kmeans_nclusters()      const {return vx_kmeans_nclusters_;}
  bool                vx_inHTsector()            const {return vx_inHTsector_;}
  bool                vx_mergebytracks()         const {return vx_mergebytracks_;}
  float               vx_TrackMinPt()            const {return vx_TrackMinPt_;}



   //=== Debug printout
  unsigned int         debug()                   const   {return debug_;}

 
  //=== Hard-wired constants
  // EJC Check this.  Found stub at r = 109.504 with flat geometry in 81X, so increased tracker radius for now.
  double               trackerOuterRadius()      const   {return 120.2;}  // max. occuring stub radius.
  // EJC Check this.  Found stub at r = 20.664 with flat geometry in 81X, so decreased tracker radius for now.
  double               trackerInnerRadius()      const   {return  20;}  // min. occuring stub radius.
  double               trackerHalfLength()       const   {return 270.;}  // half-length  of tracker. 
  double               layerIDfromRadiusBin()    const   {return 6.;}    // When counting stubs in layers, actually histogram stubs in distance from beam-line with this bin size.
 
private:
 
  static const std::map<std::string, Algorithm> algoNameMap;

  // Parameter sets for differents types of configuration parameter.
  edm::ParameterSet    genCuts_;
  edm::ParameterSet    l1TrackDef_;
  edm::ParameterSet    trackMatchDef_;
  edm::ParameterSet    vertex_;

  // Cuts on truth tracking particles.
  double               genMinPt_;
  double               genMaxAbsEta_;
  double               genMaxVertR_;
  double               genMaxVertZ_;
  std::vector<int>     genPdgIds_;
  unsigned int         genMinStubLayers_;

  // Rules for deciding when the track-finding has found an L1 track candidate
  bool                 useLayerID_;
  bool                 reduceLayerID_;
 
  // Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
  double               minFracMatchStubsOnReco_;
  double               minFracMatchStubsOnTP_;
  unsigned int         minNumMatchLayers_;
  unsigned int         minNumMatchPSLayers_;
  bool                 stubMatchStrict_;
 
  // Track Fitting Settings
  std::vector<std::string> trackFitters_;
  double               chi2OverNdfCut_;
  bool                 detailedFitOutput_;
  unsigned int         numTrackFitIterations_;
  bool                 killTrackFitWorstHit_;
  double               generalResidualCut_;
  double               killingResidualCut_;


  // Vertex Reconstruction configuration
  Algorithm            vx_algo_;
  float                vx_distance_;
  float                vx_resolution_;
  unsigned int         vx_distanceType_;
  bool                 vx_keepOnlyPV_;
  unsigned int         vx_minTracks_;
  bool                 vx_weightedmean_;
  float                vx_chi2cut_;
  float                tdr_vx_width_;
  bool                 vx_local_;
  float                vx_merge_distance_;
  bool                 vx_inHTsector_;
  bool                 vx_mergebytracks_;
  float                vx_TrackMinPt_;
  float                vx_dbscan_pt_;
  float                vx_dbscan_mintracks_;
  unsigned int         vx_kmeans_iterations_;
  unsigned int         vx_kmeans_nclusters_;
  
  // Debug printout
  unsigned int         debug_;
};

} // end namespace l1tVertexFinder

#endif

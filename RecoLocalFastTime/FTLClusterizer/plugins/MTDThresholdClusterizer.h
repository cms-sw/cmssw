#ifndef RecoLocalTracker_MTDClusterizer_MTDThresholdClusterizer_H
#define RecoLocalTracker_MTDClusterizer_MTDThresholdClusterizer_H

//-----------------------------------------------------------------------
//! \class MTDThresholdClusterizer
//! \brief An explicit threshold-based clustering algorithm.
//!
//! A threshold-based clustering algorithm which clusters FTLRecHits
//! into FTLClusters for each DetUnit.  The algorithm is straightforward 
//! and purely topological: the clustering process starts with seed hits
//! and continues by adding adjacent hits above the hit threshold.
//! Once the cluster is made, it has to be above the cluster threshold as 
//! well.
//! 
//! The clusterization is performed on a matrix with size
//! equal to the size of the MTD detector, each cell containing 
//! the cahrge and time of the corresponding hit
//! The matrix is reset after each clusterization.
//!
//! The search starts from seed hits, i.e. hits with sufficiently
//! large amplitudes
//!
//! FTLCluster contains a barycenter, but it should be noted that that
//! information is largely useless.  One must use a PositionEstimator
//! class to compute the RecHit position and its error for every given 
//! cluster.
//!
//! Sets the MTDArrayBuffer dimensions and pixel thresholds.
//! Makes clusters and stores them in theCache if the option
//! useCache has been set.
//-----------------------------------------------------------------------

// Base class, defines FTLRecHit and FLTCluster.  The latter includes
// FTLHit, FTLHitPos and Shift as inner classes.
//
#include "MTDClusterizerBase.h"

// The private pixel buffer
#include "MTDArrayBuffer.h"

// Parameter Set:
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>

#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"

class MTDThresholdClusterizer : public MTDClusterizerBase {
 public:

  MTDThresholdClusterizer(edm::ParameterSet const& conf);
  ~MTDThresholdClusterizer() override;

  // Full I/O in DetSet
  void clusterize( const FTLRecHitCollection & input,	
		   const MTDGeometry * geom,
		   FTLClusterCollection& output) override; 

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  //! Data storage
  MTDArrayBuffer               theBuffer;         // internal nrow * ncol matrix
  bool                             bufferAlreadySet;  // status of the buffer array
  std::vector<FTLCluster::FTLHitPos>  theSeeds;          // cached seed pixels
  std::vector<FTLCluster>            theClusters;       // resulting clusters  
  
  //! Clustering-related quantities:
  float theHitThreshold;    // Hit threshold 
  float theSeedThreshold;     // MTD cluster seed 
  float theClusterThreshold;  // Cluster threshold 

  //! Geometry-related information
  int  theNumOfRows;
  int  theNumOfCols;

  DetId currentId;

  bool setup(const MTDGeometry * geometry, const DetId& id);
  void copy_to_buffer( RecHitIterator itr);   
  void clear_buffer( RecHitIterator itr);
  FTLCluster make_cluster( const FTLCluster::FTLHitPos& hit );
};

#endif

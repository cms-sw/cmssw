#ifndef RecoLocalTracker_SiPixelClusterizer_PixelThresholdClusterizerForBricked_H
#define RecoLocalTracker_SiPixelClusterizer_PixelThresholdClusterizerForBricked_H

//-----------------------------------------------------------------------
//! \class PixelThresholdClusterizerForBricked
//! \brief An explicit threshold-based clustering algorithm.
//!
//! A threshold-based clustering algorithm which clusters SiPixelDigis
//! into SiPixelClusters for each DetUnit.  The algorithm is straightforward
//! and purely topological: the clustering process starts with seed pixels
//! and continues by adding adjacent pixels above the pixel threshold.
//! Once the cluster is made, it has to be above the cluster threshold as
//! well.
//!
//! The clusterization is performed on a matrix with size
//! equal to the size of the pixel detector, each cell containing
//! the ADC count of the corresponding pixel.
//! The matrix is reset after each clusterization.
//!
//! The search starts from seed pixels, i.e. pixels with sufficiently
//! large amplitudes, found at the time of filling of the matrix
//! and stored in a
//!
//! At this point the noise and dead channels are ignored, but soon they
//! won't be.
//!
//! SiPixelCluster contains a barrycenter, but it should be noted that that
//! information is largely useless.  One must use a PositionEstimator
//! class to compute the RecHit position and its error for every given
//! cluster.
//!
//! \author Largely copied from NewPixelClusterizer in ORCA written by
//!     Danek Kotlinski (PSI).   Ported to CMSSW by Petar Maksimovic (JHU).
//!     DetSetVector data container implemented by V.Chiochia (Uni Zurich)
//!
//! Sets the PixelArrayBuffer dimensions and pixel thresholds.
//! Makes clusters and stores them in theCache if the option
//! useCache has been set.
//-----------------------------------------------------------------------

// Base class, defines SiPixelDigi and SiPixelCluster.  The latter includes
// Pixel, PixelPos and Shift as inner classes.
//
#include "PixelThresholdClusterizer.h"

class dso_hidden PixelThresholdClusterizerForBricked final : public PixelThresholdClusterizer {
public:
  PixelThresholdClusterizerForBricked(edm::ParameterSet const& conf);
  ~PixelThresholdClusterizerForBricked() override;

  // Full I/O in DetSet
  void clusterizeDetUnit(const edm::DetSet<PixelDigi>& input,
                         const PixelGeomDetUnit* pixDet,
                         const TrackerTopology* tTopo,
                         const std::vector<short>& badChannels,
                         edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) override {
    clusterizeDetUnitT(input, pixDet, tTopo, badChannels, output);
  }
  void clusterizeDetUnit(const edmNew::DetSet<SiPixelCluster>& input,
                         const PixelGeomDetUnit* pixDet,
                         const TrackerTopology* tTopo,
                         const std::vector<short>& badChannels,
                         edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) override {
    clusterizeDetUnitT(input, pixDet, tTopo, badChannels, output);
  }

private:
  template <typename T>
  void clusterizeDetUnitT(const T& input,
                          const PixelGeomDetUnit* pixDet,
                          const TrackerTopology* tTopo,
                          const std::vector<short>& badChannels,
                          edmNew::DetSetVector<SiPixelCluster>::FastFiller& output);

  SiPixelCluster make_cluster_bricked(const SiPixelCluster::PixelPos& pix,
                                      edmNew::DetSetVector<SiPixelCluster>::FastFiller& output,
                                      bool isbarrel);
};

#endif

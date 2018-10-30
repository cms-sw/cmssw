#ifndef RecoLocalTracker_SiPixelClusterizer_PixelThresholdClusterizer_H
#define RecoLocalTracker_SiPixelClusterizer_PixelThresholdClusterizer_H

//-----------------------------------------------------------------------
//! \class PixelThresholdClusterizer
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
#include "DataFormats/Common/interface/DetSetVector.h"
#include "PixelClusterizerBase.h"

// The private pixel buffer
#include "SiPixelArrayBuffer.h"

// Parameter Set:
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>


class dso_hidden PixelThresholdClusterizer final : public PixelClusterizerBase {
 public:

  PixelThresholdClusterizer(edm::ParameterSet const& conf);
  ~PixelThresholdClusterizer() override;

  // Full I/O in DetSet
  void clusterizeDetUnit( const edm::DetSet<PixelDigi> & input,	
				  const PixelGeomDetUnit * pixDet,
				  const TrackerTopology* tTopo,
				  const std::vector<short>& badChannels,
				  edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) override { clusterizeDetUnitT(input, pixDet, tTopo, badChannels, output); }
  void clusterizeDetUnit( const edmNew::DetSet<SiPixelCluster> & input,
                          const PixelGeomDetUnit * pixDet,
                          const TrackerTopology* tTopo,
                          const std::vector<short>& badChannels,
                          edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) override { clusterizeDetUnitT(input, pixDet, tTopo, badChannels, output); }

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  template<typename T>
  void clusterizeDetUnitT( const T & input,
                           const PixelGeomDetUnit * pixDet,
			   const TrackerTopology* tTopo,
                           const std::vector<short>& badChannels,
                           edmNew::DetSetVector<SiPixelCluster>::FastFiller& output);

  //! Data storage
  SiPixelArrayBuffer               theBuffer;         // internal nrow * ncol matrix
  std::vector<SiPixelCluster::PixelPos>  theSeeds;          // cached seed pixels
  std::vector<SiPixelCluster>            theClusters;       // resulting clusters  
  
  //! Clustering-related quantities:
  float thePixelThresholdInNoiseUnits;    // Pixel threshold in units of noise
  float theSeedThresholdInNoiseUnits;     // Pixel cluster seed in units of noise
  float theClusterThresholdInNoiseUnits;  // Cluster threshold in units of noise

  const int thePixelThreshold;  // Pixel threshold in electrons
  const int theSeedThreshold;   // Seed threshold in electrons
  const int theClusterThreshold;    // Cluster threshold in electrons
  const int theClusterThreshold_L1; // Cluster threshold in electrons for Layer 1
  const int theConversionFactor;    // adc to electron conversion factor
  const int theConversionFactor_L1; // adc to electron conversion factor for Layer 1
  const int theOffset;              // adc to electron conversion offset
  const int theOffset_L1;           // adc to electron conversion offset for Layer 1

  const double theElectronPerADCGain;  //  ADC to electrons conversion

  const bool doPhase2Calibration;    // The ADC --> electrons calibration is for phase-2 tracker
  const int  thePhase2ReadoutMode;   // Readout mode of the phase-2 IT digitizer
  const double thePhase2DigiBaseline;// Threshold above which digis are measured in the phase-2 IT
  const int  thePhase2KinkADC;       // ADC count at which the kink in the dual slop kicks in

  //! Geometry-related information
  int  theNumOfRows;
  int  theNumOfCols;
  uint32_t theDetid;
  int theLayer;
  const bool doMissCalibrate; // Use calibration or not
  const bool doSplitClusters;
  //! Private helper methods:
  bool setup(const PixelGeomDetUnit * pixDet);
  void copy_to_buffer( DigiIterator begin, DigiIterator end );   
  void copy_to_buffer( ClusterIterator begin, ClusterIterator end );
  void clear_buffer( DigiIterator begin, DigiIterator end );
  void clear_buffer( ClusterIterator begin, ClusterIterator end );
  SiPixelCluster make_cluster( const SiPixelCluster::PixelPos& pix, edmNew::DetSetVector<SiPixelCluster>::FastFiller& output);
  // Calibrate the ADC charge to electrons 
  int calibrate(int adc, int col, int row);

};

#endif

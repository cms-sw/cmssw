#ifndef RecoLocalTracker_Phase2ITPixelClusterizer_Phase2ITPixelThresholdClusterizer_H
#define RecoLocalTracker_Phase2ITPixelClusterizer_Phase2ITPixelThresholdClusterizer_H

//-----------------------------------------------------------------------
//! \class Phase2ITPixelThresholdClusterizer
//! \brief An explicit threshold-based clustering algorithm.
//!
//! A threshold-based clustering algorithm which clusters SiPixelDigis 
//! into Phase2ITPixelClusters for each DetUnit.  The algorithm is straightforward 
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
//! Phase2ITPixelCluster contains a barycenter, but it should be noted that 
//! information is largely useless.  One must use a PositionEstimator
//! class to compute the RecHit position and its error for every given 
//! cluster.
//!
//! \author Original code by  Danek Kotlinski (PSI).   
//!         Ported to CMSSW by Petar Maksimovic (JHU).
//!         DetSetVector data container implemented by V.Chiochia (Uni Zurich)
//!
//! Sets the Phase2ITPixelArrayBuffer dimensions and pixel thresholds.
//! Makes clusters and stores them in theCache if the option
//! useCache has been set.
//-----------------------------------------------------------------------

// Base class, defines SiPixelDigi and Phase2ITPixelCluster.  The latter includes
// Pixel, PixelPos and Shift as inner classes.
//
#include "DataFormats/Common/interface/DetSetVector.h"
#include "Phase2ITPixelClusterizerBase.h"

// The private pixel buffer
#include "Phase2ITPixelArrayBuffer.h"

// Parameter Set:
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>


class dso_hidden Phase2ITPixelThresholdClusterizer final : public Phase2ITPixelClusterizerBase {
 public:

  Phase2ITPixelThresholdClusterizer(edm::ParameterSet const& conf);
  ~Phase2ITPixelThresholdClusterizer();

  // Full I/O in DetSet
  void clusterizeDetUnit( const edm::DetSet<PixelDigi> & input,	
				  const PixelGeomDetUnit * pixDet,
				  const std::vector<short>& badChannels,
				  edmNew::DetSetVector<Phase2ITPixelCluster>::FastFiller& output
);

  
 private:

  edm::ParameterSet conf_;

  //! Data storage
  Phase2ITPixelArrayBuffer           theBuffer;         // internal nrow * ncol matrix
  bool                             bufferAlreadySet;  // status of the buffer array
  std::vector<Phase2ITPixelCluster::PixelPos>  theSeeds;          // cached seed pixels
  std::vector<Phase2ITPixelCluster>            theClusters;       // resulting clusters  
  
  //! Clustering-related quantities:
  float thePixelThresholdInNoiseUnits;    // Pixel threshold in units of noise
  float theSeedThresholdInNoiseUnits;     // Pixel cluster seed in units of noise
  float theClusterThresholdInNoiseUnits;  // Cluster threshold in units of noise

  int   thePixelThreshold;  // Pixel threshold in electrons
  int   theSeedThreshold;   // Seed threshold in electrons 
  float theClusterThreshold;  // Cluster threshold in electrons
  int   theConversionFactor;  // adc to electron conversion factor
  int   theOffset;            // adc to electron conversion offset

  //! Geometry-related information
  int  theNumOfRows;
  int  theNumOfCols;
  uint32_t detid_;
  bool dead_flag;
  bool doMissCalibrate; // Use calibration or not
  bool doSplitClusters;
  //! Private helper methods:
  bool setup(const PixelGeomDetUnit * pixDet);
  void copy_to_buffer( DigiIterator begin, DigiIterator end );   
  void clear_buffer( DigiIterator begin, DigiIterator end );   
  Phase2ITPixelCluster make_cluster( const Phase2ITPixelCluster::PixelPos& pix, edmNew::DetSetVector<Phase2ITPixelCluster>::FastFiller& output
);
  // Calibrate the ADC charge to electrons 
  int calibrate(int adc, int col, int row);
  int   theStackADC_;          // The maximum ADC count for the stack layers
  int   theFirstStack_;        // The index of the first stack layer


};

#endif

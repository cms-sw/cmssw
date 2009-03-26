#ifndef RecoLocalTracker_SiStripClusterizer_SiStripClusterizerAlgo_H
#define RecoLocalTracker_SiStripClusterizer_SiStripClusterizerAlgo_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include <vector>

namespace sistrip { class RawToClustersLazyUnpacker; }

/**
   @author M.Wingham, R.Bainbridge
   @class SiStripClusterizerAlgo
   @brief Abstract base class for clusterizer algorithms
*/
class SiStripClusterizerAlgo {
  
  // Access to private methods for RawToCluster code
  friend class OldSiStripRawToClustersLazyUnpacker;
  friend class sistrip::RawToClustersLazyUnpacker;
  friend class StripByStripTestDriver;
  
 public:

  typedef std::vector<SiStripDigi> DigisV;
  typedef edm::DetSet<SiStripDigi> DigisDS;
  typedef edmNew::DetSet<SiStripDigi> DigisDSnew;
  typedef edm::DetSetVector<SiStripDigi> DigisDSV;
  typedef edmNew::DetSetVector<SiStripDigi> DigisDSVnew;
  
  typedef std::vector<SiStripCluster> ClustersV;
  typedef edm::DetSet<SiStripCluster> ClustersDS;
  typedef edm::DetSetVector<SiStripCluster> ClustersDSV;
  typedef edmNew::DetSetVector<SiStripCluster> ClustersDSVnew;
  
  /// Constructor
  SiStripClusterizerAlgo( const edm::ParameterSet& );
  
  /// Virtual destructor
  virtual ~SiStripClusterizerAlgo();
  
  /// Digis (new DSV) to Clusters (new DSV)
  void clusterize( const DigisDSVnew&, ClustersDSVnew& );
  
  /// Digis (old DSV) to Clusters (new DSV)
  void clusterize( const DigisDSV&, ClustersDSVnew& );
  
  /// Digis (old DSV) to Clusters (old DSV)
  void clusterize( const DigisDSV&, ClustersDSV& );
  
  /// Provides access to calibration constants for algorithms
  void eventSetup( const edm::EventSetup& );
  
 protected:

  /// Pure virtual method to be implemented in derived class
  virtual void clusterize( const DigisDSnew&, ClustersV& ) = 0;
  
  /// Check DetId against cached value
  bool checkDetId( uint32_t det_id ) const;
  
  /// Cache conditions data
  void conditions( uint32_t det_id );

  /** Access to noise object for algorithms. */
  inline const SiStripNoises* const noise() const;

  /** Access to quality object for algorithms. */
  inline const SiStripQuality* const quality() const;

  /** Access to gain object for algorithms. */
  inline const SiStripGain* const gain() const;

  /// Access to noise range for algorithms
  SiStripNoises::Range noiseRange() const;
  
  /// Access to quality range for algorithms
  SiStripQuality::Range qualityRange() const;
  
  /// Access to gain range for algorithms
  SiStripApvGain::Range gainRange() const;
  
 private:

  /// Private default constructor
  SiStripClusterizerAlgo() {;}

  // Building of clusters on strip-by-strip basis
  virtual void add( ClustersV&,
		    const uint32_t& id,
		    const uint16_t& strip,
		    const uint16_t& adc ) {;}

  virtual void endDet( ClustersV&, const uint32_t& id ) {;}
  
  const SiStripNoises*  noise_;
  const SiStripQuality* quality_;
  const SiStripGain*    gain_;
  
  uint32_t nCacheId_;
  uint32_t qCacheId_;
  uint32_t gCacheId_;
  
  SiStripNoises::Range  noiseRange_;
  SiStripQuality::Range qualityRange_;
  SiStripApvGain::Range gainRange_;
  
  uint32_t detId_;
  
};

inline bool SiStripClusterizerAlgo::checkDetId( uint32_t det_id ) const { return ( det_id == detId_ ); }
const SiStripNoises* const SiStripClusterizerAlgo::noise() const { return noise_; }
const SiStripQuality* const SiStripClusterizerAlgo::quality() const { return quality_; }
const SiStripGain* const SiStripClusterizerAlgo::gain() const { return gain_; }
inline SiStripNoises::Range SiStripClusterizerAlgo::noiseRange() const { return noiseRange_; }
inline SiStripQuality::Range SiStripClusterizerAlgo::qualityRange() const { return qualityRange_; }
inline SiStripApvGain::Range SiStripClusterizerAlgo::gainRange() const { return gainRange_; }

#endif // RecoLocalTracker_SiStripClusterizer_SiStripClusterizerAlgo_H




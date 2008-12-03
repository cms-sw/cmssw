#ifndef RecoLocalTracker_SiStripClusterizer_SiStripClusterizerAlgo_H
#define RecoLocalTracker_SiStripClusterizer_SiStripClusterizerAlgo_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

namespace sistrip { class RawToClustersLazyUnpacker; }

/**
   @author M.Wingham, D.Giordano, R.Bainbridge
   @class SiStripClusterizerAlgo
   @brief Abstract base class for clusterizer algorithms
*/
class SiStripClusterizerAlgo {

  /** Access to private add() method for RawToCluster. */
  friend class SiStripRawToClustersLazyUnpacker;
  friend class sistrip::RawToClustersLazyUnpacker;

 public:

  /** Class constructor. */
  SiStripClusterizerAlgo( const edm::ParameterSet& );

  /** Virtual constructor. */
  virtual ~SiStripClusterizerAlgo();

  virtual void clusterize( const edm::DetSet<SiStripDigi>&,
			   edm::DetSetVector<SiStripCluster>& ) = 0;
  
  /** Provides access to calibration constants for algorithms. */
  void eventSetup( const edm::EventSetup& );
  
 protected:
  
  /** Access to noise for algorithms. */
  inline const SiStripNoises* const noise();

  /** Access to quality for algorithms. */
  inline const SiStripQuality* const quality();

  /** Access to gain for algorithms. */
  inline const SiStripGain* const gain();
  
 private:

  /** Private default constructor. */
  SiStripClusterizerAlgo() {;}

  /** Building of clusters on strip-by-strip basis. */
  virtual void add( std::vector<SiStripCluster>&,
		    const uint32_t& id,
		    const uint16_t& strip,
		    const uint16_t& adc ) {;}

  virtual void endDet( std::vector<SiStripCluster>&,
		       const uint32_t& id ) {;}
  
  const SiStripNoises* noise_;
  const SiStripQuality* quality_;
  const SiStripGain* gain_;

  uint32_t nCacheId_;
  uint32_t qCacheId_;
  uint32_t gCacheId_;

};

const SiStripNoises* const SiStripClusterizerAlgo::noise() { return noise_; }
const SiStripQuality* const SiStripClusterizerAlgo::quality() { return quality_; }
const SiStripGain* const SiStripClusterizerAlgo::gain() { return gain_; }

#endif // RecoLocalTracker_SiStripClusterizer_SiStripClusterizerAlgo_H




#ifndef RecoLocalTracker_SiStripClusterizer_SiStripThreeThresholdAlgo_H
#define RecoLocalTracker_SiStripClusterizer_SiStripThreeThresholdAlgo_H

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgo.h"

/**
   @author M.Wingham, R.Bainbridge
   @class SiStripThreeThresholdAlgo
   @brief Clusterizer algorithm
*/

class SiStripThreeThresholdAlgo : public SiStripClusterizerAlgo {
  
 public:
  
  SiStripThreeThresholdAlgo( const edm::ParameterSet& );
  
  ~SiStripThreeThresholdAlgo();
  
 private:
  
  // Clusterize at detector-level
  void clusterize( const DigisDSnew&, ClustersV& );
  
  // Build clusters on strip-by strip basis
  void add( ClustersV& data, 
	    const uint32_t& id, 
	    const uint16_t& strip, 
	    const uint16_t& adc );
  
  // Close clusters for this det
  void endDet( ClustersV&, const uint32_t& );
  
  // internal methods
  void strip( const uint16_t&, const uint16_t&, const double&, const double& );
  void pad( const uint16_t&, const uint16_t& );
  void endCluster( ClustersV&, const uint32_t& );
  bool proximity( const uint16_t& ) const;
  bool threshold( const uint16_t&, const double&, const bool ) const;
  
  // Cluster thresholds
  float stripThr_; 
  float seedThr_;
  float clustThr_; 
  uint32_t maxHoles_;

  // Cluster variables
  float charge_;
  bool seed_;
  float sigmanoise2_;
  uint16_t first_;
  std::vector<uint16_t> amps_;

  // Detector variables
  uint16_t strip_;  

  // Record of dead digis
  std::vector<uint16_t> digis_;

};

#endif // RecoLocalTracker_SiStripClusterizer_SiStripThreeThresholdAlgo_H




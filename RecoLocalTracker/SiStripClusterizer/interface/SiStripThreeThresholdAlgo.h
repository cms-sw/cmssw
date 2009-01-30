#ifndef RecoLocalTracker_SiStripClusterizer_SiStripThreeThresholdAlgo_H
#define RecoLocalTracker_SiStripClusterizer_SiStripThreeThresholdAlgo_H

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgo.h"

/**
   @author M.Wingham, D.Giordano, R.Bainbridge
   @class SiStripThreeThresholdAlgo
   @brief Clusterizer algorithm
*/

class SiStripThreeThresholdAlgo : public SiStripClusterizerAlgo {
  
 public:
  
  SiStripThreeThresholdAlgo( const edm::ParameterSet& );
  
  virtual ~SiStripThreeThresholdAlgo();
  
  virtual void clusterize( const edm::DetSet<SiStripDigi>&, edm::DetSetVector<SiStripCluster>& );
  
 private:
  
  /// Building of clusters on strip-by-strip basis.

  virtual void add( std::vector<SiStripCluster>& data, const uint32_t& id, const uint16_t& strip, const uint16_t& adc );
  
  /// close clusters for this det

  virtual void endDet( std::vector<SiStripCluster>&, const uint32_t& );
  
  /// internal methods

  void strip(const uint16_t&, const uint16_t&, const double&, const double&);
  void pad(const uint16_t&, const uint16_t&);
  void endCluster(std::vector<SiStripCluster>&, const uint32_t&);
  bool proximity(const uint16_t&) const;
  bool threshold(const uint16_t&, const double&, const bool) const;
  
  /// cluster thresholds

  float stripThr_; 
  float seedThr_;
  float clustThr_; 
  uint32_t maxHoles_;

  /// cluster variables

  float charge_;
  bool seed_;
  float sigmanoise2_;
  uint16_t first_;
  std::vector<uint16_t> amps_;

  /// detector variables

  uint16_t strip_;  

  //Record of dead digis

  std::vector<uint16_t> digis_;
};

#endif // RecoLocalTracker_SiStripClusterizer_SiStripThreeThresholdAlgo_H




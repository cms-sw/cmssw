#ifndef RecoLocalTracker_SiStripClusterizer_SiStripDummyAlgo_H
#define RecoLocalTracker_SiStripClusterizer_SiStripDummyAlgo_H

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgo.h"

/**
   @author M.Wingham, R.Bainbridge
   @class SiStripDummyAlgo
   @brief Dummy clusterizer algorithm for testing only
*/
class SiStripDummyAlgo : public SiStripClusterizerAlgo {
  
 public:
  
  SiStripDummyAlgo( const edm::ParameterSet& );
  
  ~SiStripDummyAlgo();
  
 private:
  
  void clusterize( const DigisDS&, ClustersDS& );
  
  /// Building of clusters on strip-by-strip basis
  void add( ClustersV& data, 
	    const uint32_t& id, 
	    const uint16_t& strip, 
	    const uint16_t& adc );
  
};

#endif // RecoLocalTracker_SiStripClusterizer_SiStripDummyAlgo_H




#ifndef RecoLocalTracker_SiStripClusterizer_SiStripDummyAlgo_H
#define RecoLocalTracker_SiStripClusterizer_SiStripDummyAlgo_H

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgo.h"

/**
   @author M.Wingham, D.Giordano, R.Bainbridge
   @class SiStripDummyAlgo
   @brief Dummy clusterizer algorithm for testing only
*/
class SiStripDummyAlgo : public SiStripClusterizerAlgo {
  
 public:
  
  SiStripDummyAlgo( const edm::ParameterSet& );
  
  virtual ~SiStripDummyAlgo();
  
  virtual void clusterize( const edm::DetSet<SiStripDigi>&,
			   edm::DetSetVector<SiStripCluster>& );
  
 private:
  
  /** Building of clusters on strip-by-strip basis. */
  virtual void add( edm::DetSet<SiStripCluster>&,
		    const uint16_t& strip,
		    const uint16_t& adc );
  
};

#endif // RecoLocalTracker_SiStripClusterizer_SiStripDummyAlgo_H




#ifndef RecoLocalTracker_SiStripClusterizer_SiStripFrontEndDriverAlgo_H
#define RecoLocalTracker_SiStripClusterizer_SiStripFrontEndDriverAlgo_H

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgo.h"

/**
   @author M.Wingham, R.Bainbridge
   @class SiStripFrontEndDriverAlgo
   @brief Clusterizer algorithm replicating Front-End Driver
*/
class SiStripFrontEndDriverAlgo : public SiStripClusterizerAlgo {
  
 public:
  
  SiStripFrontEndDriverAlgo( const edm::ParameterSet& );
  
  ~SiStripFrontEndDriverAlgo();
  
  void clusterize( const DigisDS&, ClustersDS& );
  
 private:
  
  // Build clusters on strip-by strip basis
  void add( ClustersV& data, 
	    const uint32_t& id, 
	    const uint16_t& strip, 
	    const uint16_t& adc );
  
};

#endif // RecoLocalTracker_SiStripClusterizer_SiStripFrontEndDriverAlgo_H




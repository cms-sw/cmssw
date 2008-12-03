#ifndef RecoLocalTracker_SiStripClusterizer_SiStripFrontEndDriverAlgo_H
#define RecoLocalTracker_SiStripClusterizer_SiStripFrontEndDriverAlgo_H

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgo.h"

/**
   @author M.Wingham, D.Giordano, R.Bainbridge
   @class SiStripFrontEndDriverAlgo
   @brief Clusterizer algorithm replicating Front-End Driver
*/
class SiStripFrontEndDriverAlgo : public SiStripClusterizerAlgo {
  
 public:
  
  SiStripFrontEndDriverAlgo( const edm::ParameterSet& );
  
  virtual ~SiStripFrontEndDriverAlgo();
  
  virtual void clusterize( const edm::DetSet<SiStripDigi>&,
			   edm::DetSetVector<SiStripCluster>& );
  
 private:
  
  /** Building of clusters on strip-by-strip basis. */
  virtual void add( edm::DetSet<SiStripCluster>&,const uint16_t& strip,const uint16_t& adc );
  
};

#endif // RecoLocalTracker_SiStripClusterizer_SiStripFrontEndDriverAlgo_H




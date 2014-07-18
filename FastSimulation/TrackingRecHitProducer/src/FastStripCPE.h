#ifndef FastSimulation_TrackingRecHitProducer_FastStripCPE_H
#define FastSimulation_TrackingRecHitProducer_FastStripCPE_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <ext/hash_map>
#include <map>
#include <memory>

class FastStripCPE : public StripClusterParameterEstimator 
{
 public:
  FastStripCPE(){;}
  
  //Standard method used
  //LocalValues is typedef for std::pair<LocalPoint,LocalError> 
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl,const GeomDetUnit& det) const override;  


  LocalVector driftDirection(const StripGeomDetUnit* ) const override;
  
};

#endif





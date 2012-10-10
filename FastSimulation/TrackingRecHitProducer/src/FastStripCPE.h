#ifndef FastSimulation_TrackingRecHitProducer_FastStripCPE_H
#define FastSimulation_TrackingRecHitProducer_FastStripCPE_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <ext/hash_map>
#include <map>


class FastStripCPE : public StripClusterParameterEstimator 
{
 public:
  FastStripCPE(){;}
  
  //Standard method used
  //LocalValues is typedef for std::pair<LocalPoint,LocalError> 
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl,const GeomDetUnit& det) const {
    return localParameters(cl);
  }; 
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl)const; 
  
  //Put information into the map.
  void enterLocalParameters(uint32_t id, uint16_t firstStrip, std::pair<LocalPoint,LocalError> pos_err_info) const;
  
  //Clear the map.
  void clearParameters() const {
    pos_err_map.clear();
  }
  
  LocalVector driftDirection(const StripGeomDetUnit* det)const;
  
 private:
  mutable std::map<std::pair<uint32_t, uint16_t>,std::pair<LocalPoint, LocalError> >  pos_err_map;
  
};

#endif





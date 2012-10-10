#ifndef FastSimulation_TrackingRecHitProducer_FastPixelCPE_H
#define FastSimulation_TrackingRecHitProducer_FastPixelCPE_H

//Header files
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "FastSimDataFormats/External/interface/FastTrackerCluster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <map>

#if 0
#endif

class FastPixelCPE : public PixelClusterParameterEstimator
{
 public:
  FastPixelCPE(){;}
  
  //Standard method used
  //LocalValues is typedef for std::pair<LocalPoint,LocalError> 
  PixelClusterParameterEstimator::LocalValues localParameters( const SiPixelCluster & cl,
							       const GeomDetUnit    & det) const;
	
  //Put information into the map.
  void enterLocalParameters(unsigned int id, std::pair<int,int> &row_col, std::pair<LocalPoint,LocalError> pos_err_info) const; 
  
  //Clear the map.
  void clearParameters() const { 
    pos_err_map.clear(); 
  }
  
 private:
  //Map used to store clusters distinctly.
  mutable std::map<std::pair<unsigned int, std::pair<int,int> >, std::pair<LocalPoint, LocalError> > pos_err_map;
};

#endif





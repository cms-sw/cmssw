#ifndef RecoLocalTracker_Cluster_Parameter_Estimator_H
#define RecoLocalTracker_Cluster_Parameter_Estimator_H

#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

template <class T> class ClusterParameterEstimator {
  
 public:
   typedef std::pair<LocalPoint,LocalError>  LocalValues; 
   virtual LocalValues localParameters( const T&,const GeomDetUnit&) const = 0; 
   virtual LocalValues localParameters( const T& cluster, const GeomDetUnit& gd, float alpha, float beta) const {
     return localParameters(cluster,gd);
   } 

  virtual ~ClusterParameterEstimator(){}
  
};

#endif

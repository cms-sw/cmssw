#ifndef RecoLocalTracker_PixelCluster_Parameter_Estimator_H
#define RecoLocalTracker_PixelCluster_Parameter_Estimator_H

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"
#include<tuple>

class PixelClusterParameterEstimator
{
  public:

  virtual ~PixelClusterParameterEstimator(){}
  
  typedef std::pair<LocalPoint,LocalError>  LocalValues;
  typedef std::vector<LocalValues> VLocalValues;

  using ReturnType = std::tuple<LocalPoint,LocalError,SiPixelRecHitQuality::QualWordType>;

  // here just to implement it in the clients;
  // to be properly implemented in the sub-classes in order to make them thread-safe

  virtual ReturnType getParameters(const SiPixelCluster & cl, 
                                   const GeomDetUnit    & det) const =0;

  virtual ReturnType getParameters(const SiPixelCluster & cl, 
				   const GeomDetUnit    & det, 
				   const LocalTrajectoryParameters & ltp ) const =0;

  virtual ReturnType getParameters(const SiPixelCluster & cl, 
				   const GeomDetUnit    & det, 
				   const TrajectoryStateOnSurface& tsos ) const {
    return getParameters(cl,det,tsos.localParameters());
  }

  virtual VLocalValues localParametersV(const SiPixelCluster& cluster, const GeomDetUnit& gd) const {
    VLocalValues vlp;
    ReturnType tuple = getParameters(cluster, gd);
    vlp.push_back(std::make_pair(std::get<0>(tuple), std::get<1>(tuple)));
    return vlp;
  }
  virtual VLocalValues localParametersV(const SiPixelCluster& cluster, const GeomDetUnit& gd, TrajectoryStateOnSurface& tsos) const {
    VLocalValues vlp;
    ReturnType tuple = getParameters(cluster,  gd, tsos);
    vlp.push_back(std::make_pair(std::get<0>(tuple), std::get<1>(tuple)));
    return vlp;
  }


  PixelClusterParameterEstimator() : clusterProbComputationFlag_(0){}

  //--- Flag to control how SiPixelRecHits compute clusterProbability().
  //--- Note this is set via the configuration file, and it's simply passed
  //--- to each TSiPixelRecHit.
  inline unsigned int clusterProbComputationFlag() const 
    { 
      return clusterProbComputationFlag_ ; 
    }
  
  
  protected:
 //--- A flag that could be used to change the behavior of
  //--- clusterProbability() in TSiPixelRecHit (the *transient* one).
  //--- The problem is that the transient hits are made after the CPE runs
  //--- and they don't get the access to the PSet, so we pass it via the
  //--- CPE itself...
  //
  unsigned int clusterProbComputationFlag_;

};

#endif

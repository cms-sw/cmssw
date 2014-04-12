#ifndef RecoLocalTracker_PixelCluster_Parameter_Estimator_H
#define RecoLocalTracker_PixelCluster_Parameter_Estimator_H

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"
#include<tuple>

class PixelClusterParameterEstimator : public  ClusterParameterEstimator<SiPixelCluster> {
  public:

  
  using ReturnType = std::tuple<LocalPoint,LocalError,SiPixelRecHitQuality::QualWordType>;

  // here just to implement it in the clients;
  // to be properly implemented in the sub-classes in order to make them thread-safe
  virtual ReturnType getParameters(const SiPixelCluster & cl, 
				   const GeomDetUnit    & det ) const {
    auto lv = localParameters(cl,det);
    return std::make_tuple(lv.first,lv.second,rawQualityWord());
  }
  virtual ReturnType getParameters(const SiPixelCluster & cl, 
				   const GeomDetUnit    & det, 
				   const LocalTrajectoryParameters & ltp ) const {
    auto lv = localParameters(cl,det,ltp);
    return std::make_tuple(lv.first,lv.second,rawQualityWord());
  }



  PixelClusterParameterEstimator() : clusterProbComputationFlag_(0){}

  //--- Flag to control how SiPixelRecHits compute clusterProbability().
  //--- Note this is set via the configuration file, and it's simply passed
  //--- to each TSiPixelRecHit.
  inline unsigned int clusterProbComputationFlag() const 
    { 
      return clusterProbComputationFlag_ ; 
    }
  
  
  //-----------------------------------------------------------------------------
  //! A convenience method to fill a whole SiPixelRecHitQuality word in one shot.
  //! This way, we can keep the details of what is filled within the pixel
  //! code and not expose the Transient SiPixelRecHit to it as well.  The name
  //! of this function is chosen to match the one in SiPixelRecHit.
  //-----------------------------------------------------------------------------
  virtual SiPixelRecHitQuality::QualWordType rawQualityWord() const {
     return SiPixelRecHitQuality::QualWordType();
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

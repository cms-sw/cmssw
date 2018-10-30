#ifndef RecoLocalTracker_Fake_PixelCluster_Parameter_Estimator_H
#define RecoLocalTracker_Fake_PixelCluster_Parameter_Estimator_H

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"
#include<tuple>


#include "RecoLocalTracker/ClusterParameterEstimator/interface/FakeCPE.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

class PixelFakeCPE final : public PixelClusterParameterEstimator {
public:

  PixelFakeCPE() = default;
  ~PixelFakeCPE() = default;

  typedef std::pair<LocalPoint,LocalError>  LocalValues;
  typedef std::vector<LocalValues> VLocalValues;

  using ReturnType = std::tuple<LocalPoint,LocalError,SiPixelRecHitQuality::QualWordType>;

  // here just to implement it in the clients;
  // to be properly implemented in the sub-classes in order to make them thread-safe

  ReturnType getParameters(const SiPixelCluster & cl, 
                                   const GeomDetUnit    & det) const override {
   auto const & lv = fakeCPE().map().get(cl,det);
   return {lv.first,lv.second,0};
  }

  ReturnType getParameters(const SiPixelCluster & cl, 
				   const GeomDetUnit    & det, 
				   const LocalTrajectoryParameters &) const override{
      return getParameters(cl,det);
   }

  void setFakeCPE(FakeCPE * iFakeCPE) { m_fakeCPE = iFakeCPE;} 
  FakeCPE const & fakeCPE() const { return *m_fakeCPE; }

private:
  FakeCPE const * m_fakeCPE=nullptr;

};

#endif

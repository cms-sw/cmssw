#ifndef RecoLocalFastTime_MTDCluster_Parameter_Estimator_H
#define RecoLocalFastTime_MTDCluster_Parameter_Estimator_H

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/FTLRecHit/interface/FTLCluster.h"

#include <tuple>

class MTDClusterParameterEstimator {
public:
  virtual ~MTDClusterParameterEstimator() {}

  typedef std::pair<LocalPoint, LocalError> LocalValues;
  typedef std::vector<LocalValues> VLocalValues;

  typedef float TimeValue;
  typedef float TimeValueError;

  using ReturnType = std::tuple<LocalPoint, LocalError, TimeValue, TimeValueError>;

  // here just to implement it in the clients;
  // to be properly implemented in the sub-classes in order to make them thread-safe

  virtual ReturnType getParameters(const FTLCluster& cl, const GeomDetUnit& det) const = 0;

  virtual ReturnType getParameters(const FTLCluster& cl,
                                   const GeomDetUnit& det,
                                   const LocalTrajectoryParameters& ltp) const = 0;

  virtual ReturnType getParameters(const FTLCluster& cl,
                                   const GeomDetUnit& det,
                                   const TrajectoryStateOnSurface& tsos) const {
    return getParameters(cl, det, tsos.localParameters());
  }

  virtual VLocalValues localParametersV(const FTLCluster& cluster, const GeomDetUnit& gd) const {
    VLocalValues vlp;
    ReturnType tuple = getParameters(cluster, gd);
    vlp.emplace_back(std::get<0>(tuple), std::get<1>(tuple));
    return vlp;
  }
  virtual VLocalValues localParametersV(const FTLCluster& cluster,
                                        const GeomDetUnit& gd,
                                        TrajectoryStateOnSurface& tsos) const {
    VLocalValues vlp;
    ReturnType tuple = getParameters(cluster, gd, tsos);
    vlp.emplace_back(std::get<0>(tuple), std::get<1>(tuple));
    return vlp;
  }

  MTDClusterParameterEstimator(){};
};

#endif

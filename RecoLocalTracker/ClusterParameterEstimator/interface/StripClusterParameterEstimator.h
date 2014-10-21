#ifndef RecoLocalTracker_StripCluster_Parameter_Estimator_H
#define RecoLocalTracker_StripCluster_Parameter_Estimator_H

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"

#include "FWCore/Utilities/interface/Exception.h"


/**
    A StripClusterParameterEstimator specific for strips
   also implements direct access to measurement frame, since that is needed during the track refitting

**/

class StripClusterParameterEstimator
{
 public:
  typedef std::pair<LocalPoint,LocalError>  LocalValues;
  typedef std::vector<LocalValues> VLocalValues;

  virtual LocalValues localParameters( const SiStripCluster&,const GeomDetUnit&) const {
      return std::make_pair(LocalPoint(), LocalError());
  }
  virtual LocalValues localParameters( const SiStripCluster& cluster, const GeomDetUnit& gd, const LocalTrajectoryParameters&) const {
    return localParameters(cluster,gd);
  }
  virtual LocalValues localParameters( const SiStripCluster& cluster, const GeomDetUnit& gd, const TrajectoryStateOnSurface& tsos) const {
    return localParameters(cluster,gd,tsos.localParameters());
  }
  virtual VLocalValues localParametersV( const SiStripCluster& cluster, const GeomDetUnit& gd) const {
    VLocalValues vlp;
    vlp.push_back(localParameters(cluster,gd));
    return vlp;
  }
  virtual VLocalValues localParametersV( const SiStripCluster& cluster, const GeomDetUnit& gd, const TrajectoryStateOnSurface& tsos) const {
    VLocalValues vlp;
    vlp.push_back(localParameters(cluster,gd,tsos.localParameters()));
    return vlp;
  }

  
  // used by Validation....
  virtual LocalVector driftDirection(const StripGeomDetUnit* ) const =0;

  virtual ~StripClusterParameterEstimator(){}
  


};


#endif





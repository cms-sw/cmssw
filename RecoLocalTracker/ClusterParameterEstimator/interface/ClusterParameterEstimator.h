#ifndef RecoLocalTracker_Cluster_Parameter_Estimator_H
#define RecoLocalTracker_Cluster_Parameter_Estimator_H

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


template <class T> 
class ClusterParameterEstimator {
  
 public:
  typedef std::pair<LocalPoint,LocalError>  LocalValues;
  typedef std::vector<LocalValues> VLocalValues;
  virtual LocalValues localParameters( const T&,const GeomDetUnit&) const = 0; 
  virtual LocalValues localParameters( const T& cluster, const GeomDetUnit& gd, const LocalTrajectoryParameters&) const {
    return localParameters(cluster,gd);
  }
  virtual LocalValues localParameters( const T& cluster, const GeomDetUnit& gd, const TrajectoryStateOnSurface& tsos) const {
    return localParameters(cluster,gd,tsos.localParameters());
  }
  virtual VLocalValues localParametersV( const T& cluster, const GeomDetUnit& gd) const {
    VLocalValues vlp;
    vlp.push_back(localParameters(cluster,gd));
    return vlp;
  }
  virtual VLocalValues localParametersV( const T& cluster, const GeomDetUnit& gd, const LocalTrajectoryParameters& ltp) const {
    VLocalValues vlp;
    vlp.push_back(localParameters(cluster,gd,ltp));
    return vlp;
  }
  virtual VLocalValues localParametersV( const T& cluster, const GeomDetUnit& gd, const TrajectoryStateOnSurface& tsos) const {
    VLocalValues vlp;
    vlp.push_back(localParameters(cluster,gd,tsos.localParameters()));
    return vlp;
  }
  
  virtual ~ClusterParameterEstimator(){}
  
  //methods needed by FastSim
  virtual void enterLocalParameters(unsigned int id, std::pair<int,int>
				    &row_col, LocalValues pos_err_info) const {}
  virtual void enterLocalParameters(uint32_t id, uint16_t firstStrip,
				    LocalValues pos_err_info) const {}
  virtual void clearParameters() const {}
  
};

#endif

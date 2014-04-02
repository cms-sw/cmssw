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
  
  virtual ~StripClusterParameterEstimator(){}
  
  //methods needed by FastSim
  virtual void enterLocalParameters(unsigned int id, std::pair<int,int>
				    &row_col, LocalValues pos_err_info) {}
  virtual void enterLocalParameters(uint32_t id, uint16_t firstStrip,
				    LocalValues pos_err_info) {}
  virtual void clearParameters() {}

  typedef std::pair<MeasurementPoint,MeasurementError>  MeasurementValues;
  virtual LocalVector driftDirection(const StripGeomDetUnit* det)const=0;

  //
  // methods to get directly the measurements
  //

  virtual MeasurementValues measurementParameters( const SiStripCluster&,const GeomDetUnit&) const
  {
    throw cms::Exception("Not implemented")
      << "StripClusterParameterEstimator::measurementParameters not yet implemented"<< std::endl;
  }

  virtual MeasurementValues measurementParameters( const SiStripCluster& cluster,
                                                   const GeomDetUnit& gd,
                                                   const LocalTrajectoryParameters & ltp) const
  {
    throw cms::Exception("Not implemented") << "StripClusterParameterEstimator::measurementParameters not yet implemented"<<
 std::endl;
  }


  float templateProbability() const
  {
    return stripCPEtemplateProbability_;
  }

  int templateQbin() const
  {
    return stripCPEtemplateQbin_;
  }

  void templateProbability( float stp )
  {
    stripCPEtemplateProbability_ = stp;
  }

  void templateQbin( int stqb )
  {
    stripCPEtemplateQbin_ = stqb;
  }

  mutable float stripCPEtemplateProbability_;
  mutable int stripCPEtemplateQbin_;

};


#endif





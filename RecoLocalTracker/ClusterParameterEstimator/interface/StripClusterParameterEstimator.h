#ifndef RecoLocalTracker_StripCluster_Parameter_Estimator_H
#define RecoLocalTracker_StripCluster_Parameter_Estimator_H

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"

#include "FWCore/Utilities/interface/Exception.h"


/**
    A ClusterParameterEstimator specific for strips
   also implements direct access to measurement frame, since that is needed during the track refitting

**/

class StripClusterParameterEstimator : public ClusterParameterEstimator<SiStripCluster>
{
 public:
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





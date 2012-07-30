#ifndef IDEALHELIXPARAMETERS_H
#define IDEALHELIXPARAMETERS_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/* 
Given a track, evaluates the status of the track finding in the XY plane 
the point of tangent of the circle to the reference point(in general the PrimaryVertex or the BeamSpot)

The approach is the following.
Given a track, 
-extract the innermomentum at the innerpoint
-evalaute the radius of curvature and the center of the circle
-solving geometrical equation, evaluate the point of tangence to the circle, starting from the PV
-evaluate the status at this point of tangence T

 */
class IdealHelixParameters{

 public:

 IdealHelixParameters():
  _magnField(0),_track(0),
    _refPoint               (math::XYZVector(0,0,0)),
    _tangentPoint           (math::XYZVector(0,0,0)),
    _MomentumAtTangentPoint(math::XYZVector(0,0,0)){};
  ~IdealHelixParameters(){};

  void setMagnField(const MagneticField* magnField){_magnField=magnField;}
  void setData(const reco::Track* track, math::XYZVector refPoint=math::XYZVector(0,0,0));
  void setData(const reco::Track* track, math::XYZPoint ref);

  bool isTangentPointDistanceLessThan(float rmax, const reco::Track* track, const math::XYZVector refPoint);

  math::XYZVector   GetCircleCenter() const {return _circleCenter;}
  math::XYZVector   GetTangentPoint() const {return _tangentPoint;}
  math::XYZVector   GetMomentumAtTangentPoint() const {return _MomentumAtTangentPoint;}
  float          GetTransverseIPAtTangent() const {return _transverseIP;}
  float          GetRotationAngle() const {return _rotationAngle;}
  
 private:

  void calculate();
  void evalCircleCenter();
  void evalTangentPoint();
  void evalMomentumatTangentPoint();

  const MagneticField *_magnField;
  const reco::Track   *_track;
  float  _radius;
  math::XYZVector  _circleCenter;
  math::XYZVector  _refPoint;
  math::XYZVector  _tangentPoint;
  math::XYZVector _MomentumAtTangentPoint;
  float _transverseIP;
  float _rotationAngle;

  //  std::stringstream ss;

};
#endif

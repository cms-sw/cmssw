#include "RecoTracker/ConversionSeedGenerators/interface/TrackManipulator.h"

void TrackManipulator::
setData(const reco::Track* track,  math::XYZPoint ref){
  setData(track,math::XYZVector(ref.x(),ref.y(),ref.z())); 
}

void TrackManipulator::
setData(const reco::Track* track,  math::XYZVector refPoint){

  if(track!=0 && _track!=track){
    _track=track;
    _refPoint=refPoint;
    calculate();
  }

  if((refPoint-_refPoint).r()!=0){
    _refPoint=refPoint;
    calculate();
  }
}


void TrackManipulator::
calculate(){

  LogDebug("TrackManipulator")
    << "[TrackManipulator] " 
    << "\n\t "<< "refPoint \t" <<_refPoint
    << "\n\t "<< "innerDetid    \t" << _track->innerDetId()
    << "\n\t "<< "innerMomentum \t" << _track->innerMomentum()
    << "\n\t "<< "innerPosition \t" << _track->innerPosition();

  evalCircleCenter();
  
  evalTangentPoint();

  evalMomentumatTangentPoint();
  

  /* DEBUG
    math::XYZVector _checkMomentumAtTangentoPoint (
    cos_angle*innerMomentum.x()+sin_angle*innerMomentum.y(), 
    -1*sin_angle*innerMomentum.x()+cos_angle*innerMomentum.y(), 
    innerMomentum.z()
    );
    math::XYZVector cp_xy(_checkMomentumAtTangentoPoint.x(),_checkMomentumAtTangentoPoint.y(),0);
    float _checktransverseIP = (r_xy.Cross(cp_xy)).z()/cp_xy.rho();

    ss 
    << "\n\t "<< "_checkMomentumAtTangentoPoint \t" <<_checkMomentumAtTangentoPoint
    << "\n\t "<< "_checktransverseIP \t" <<_checktransverseIP;
  */

}

void TrackManipulator::
evalCircleCenter(){
  GlobalPoint  innerPos(_track->innerPosition().x(),_track->innerPosition().y(),_track->innerPosition().z());
  GlobalVector innerMom(_track->innerMomentum().x(),_track->innerMomentum().y(),_track->innerMomentum().z());

  GlobalTrajectoryParameters globalTrajParam(
					     innerPos,
					     innerMom,
					     _track->charge(),
					     _magnField
					     );

  _radius = 1./fabs(globalTrajParam.transverseCurvature());
  float phi = innerMom.phi();

  //eval Center of the Circumference passing in track->innserPosition
  float Xc = innerPos.x() + _track->charge()* _radius * sin(phi);
  float Yc = innerPos.y() - _track->charge()* _radius * cos(phi);

  
  _circleCenter.SetXYZ(Xc,Yc,innerPos.z());  //NB Z component is useless

  LogDebug("TrackManipulator")
      << "\n\t "<< "circle center \t" <<_circleCenter
      << "\n\t "<< "radius "          <<_radius; 

}


void TrackManipulator::
evalMomentumatTangentPoint(){
  
  if(_tangentPoint.r()==0){
    _MomentumAtTangentPoint = math::XYZVector(0.,0.,0.);
    return;
  }

  math::XYZVector innerPosition(_track->innerPosition().X(),_track->innerPosition().Y(),_track->innerPosition().Z());
  math::XYZVector innerMomentum(_track->innerMomentum());


  math::XYZVector pCi=innerPosition-_circleCenter;
  math::XYZVector pCT=_tangentPoint-_circleCenter;

  math::XYZVector pCi_xy(pCi.x(),pCi.y(),0);
  math::XYZVector pCT_xy(pCT.x(),pCT.y(),0);
  
  float cos_angle = pCi_xy.Dot(pCT_xy)/pCi_xy.rho()/pCT_xy.rho();
  float sin_angle = pCi_xy.Cross(pCT_xy).z()/pCi_xy.rho()/pCT_xy.rho();


  _MomentumAtTangentPoint = math::XYZVector(
					    cos_angle*innerMomentum.x()-sin_angle*innerMomentum.y(), 
					    sin_angle*innerMomentum.x()+cos_angle*innerMomentum.y(), 
					    innerMomentum.z()
					    );

  math::XYZVector r_(_tangentPoint.x(),_tangentPoint.y(),0);
  math::XYZVector p_(_MomentumAtTangentPoint.x(),_MomentumAtTangentPoint.y(),0);
  
  _transverseIP = (r_.x()*p_.y()-r_.y()*p_.x())/p_.rho()/r_.rho();
  _rotationAngle=atan(sin_angle/cos_angle);


  
  LogDebug("TrackManipulator")	<< "\n\t "<< "_MomentumAtTangentPoint \t" <<_MomentumAtTangentPoint
				<< "\n\t "<< "sin_angle \t" <<sin_angle << " angle \t" <<asin(sin_angle)
				<< "\n\t "<< "cos_angle \t" <<cos_angle << " angle \t" <<acos(cos_angle)
				<< "\n\t "<< "_rotationAngle \t" <<_rotationAngle
				<< "\n\t "<< "_transverseIP \t" <<_transverseIP
				<< "\n\t "<< " check similitude pz/pt "<< _track->innerMomentum().z()/_track->innerMomentum().rho() << " pZ/pRho " << innerPosition.z()/innerPosition.rho();
}


void TrackManipulator::
evalTangentPoint(){

  math::XYZVector innerPosition(_track->innerPosition().X(),_track->innerPosition().Y(),_track->innerPosition().Z());

  math::XYZVector vL = _circleCenter-_refPoint;
  float l = vL.rho();

  if(l<_radius){
    //refpoint is inside the circle
    _tangentPoint.SetXYZ(0.,0.,0.);
    return;
  }

  float sin_alpha = _radius/l;

  //there are two possible tangents, with Point of tangence T1 and T2, and same distance T from _refPoint 
  float T=sqrt(l*l-_radius*_radius);

  math::XYZVector vLperp(-vL.y(),vL.x(),0); //NB: z component not correct
  math::XYZVector tmpT1 = (1-sin_alpha*sin_alpha)*vL + T/l*sin_alpha*vLperp;
  math::XYZVector tmpT2 = (1-sin_alpha*sin_alpha)*vL - T/l*sin_alpha*vLperp;

  if( (tmpT1-innerPosition).rho()<(tmpT2-innerPosition).rho())
    _tangentPoint=tmpT1+_refPoint;
  else
    _tangentPoint=tmpT2+_refPoint;

  //Fix the Z component
  _tangentPoint.SetZ( (_tangentPoint.rho()-_refPoint.rho()) * (_track->innerMomentum().z()/_track->innerMomentum().rho()) + _refPoint.z() );

  LogDebug("TrackManipulator")<< "\n\t "<< "tangent Point \t" <<_tangentPoint;
}


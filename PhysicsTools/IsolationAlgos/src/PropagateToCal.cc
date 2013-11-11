#include "PhysicsTools/IsolationAlgos/interface/PropagateToCal.h"

PropagateToCal::PropagateToCal(double radius, double minZ, double maxZ, bool theIgnoreMaterial)
{ 
  radius_ = radius; 
  maxZ_ = maxZ; 
  minZ_ = minZ; 
  theIgnoreMaterial_ = theIgnoreMaterial;
  if (maxZ_ < minZ_ || radius < 0.0) 
    throw cms::Exception("BadConfig") << "PropagateToCal: CalMaxZ (" 
                                      << maxZ_
				      << ") smaller than CalMinZ (" 
				      << minZ_ << ") or invalid radius ("
				      << radius_ << ").";
}


PropagateToCal::~PropagateToCal()
{
}


bool PropagateToCal::propagate(const GlobalPoint& vertex, 
  	             GlobalVector& Cand, int charge, const MagneticField * field) const
{
  ///the code is inspired by Gero's CosmicGenFilterHelix class: 
  bool result = true;
  typedef std::pair<TrajectoryStateOnSurface, double> TsosPath;

  SteppingHelixPropagator propagator(field); // should we somehow take it from ESetup???
  propagator.setMaterialMode(theIgnoreMaterial_); // no material effects if set to true
  propagator.setNoErrorPropagation(true);

  const FreeTrajectoryState fts(GlobalTrajectoryParameters(vertex, Cand, charge, field));
  const Surface::RotationType dummyRot;

  /// target cylinder, around z-axis
  Cylinder::ConstCylinderPointer theTargetCylinder = 
    Cylinder::build(radius_, Surface::PositionType(0.,0.,0.), dummyRot);

  /// plane closing cylinder at 'negative' side  
  Plane::ConstPlanePointer theTargetPlaneMin = 
    Plane::build(Surface::PositionType(0.,0.,minZ_), dummyRot);

  /// plane closing cylinder at 'positive' side  
  Plane::ConstPlanePointer theTargetPlaneMax = 
    Plane::build(Surface::PositionType(0.,0.,maxZ_), dummyRot);

  TsosPath aTsosPath(propagator.propagateWithPath(fts, *theTargetCylinder));
  if (!aTsosPath.first.isValid()) {
    result = false;
  } else if (aTsosPath.first.globalPosition().z() < theTargetPlaneMin->position().z()) {
    // If on cylinder, but outside minimum z, try minimum z-plane:
    // (Would it be possible to miss rdius on plane, but reach cylinder afterwards in z-range?
    //  No, at least not in B-field parallel to z-axis which is cylinder axis.)
    aTsosPath = propagator.propagateWithPath(fts, *theTargetPlaneMin);
    if (!aTsosPath.first.isValid()
	|| aTsosPath.first.globalPosition().perp() > theTargetCylinder->radius()) {
      result = false;
    }
  } else if (aTsosPath.first.globalPosition().z() > theTargetPlaneMax->position().z()) {
    // Analog for outside maximum z:
    aTsosPath = propagator.propagateWithPath(fts, *theTargetPlaneMax);
    if (!aTsosPath.first.isValid()
	|| aTsosPath.first.globalPosition().perp() > theTargetCylinder->radius()) {
      result = false;
    }
  }
  ///The result is the vector connecting the extrapolation endPoint on the Calorimeter surface and
  ///the origin of the coordinate system, point (0,0,0).
  if (result) {
    Cand = GlobalVector(aTsosPath.first.globalPosition().x(),
			aTsosPath.first.globalPosition().y(),
			aTsosPath.first.globalPosition().z() );
  }
  return result;///Successfully propagated to the calorimeter or not
}

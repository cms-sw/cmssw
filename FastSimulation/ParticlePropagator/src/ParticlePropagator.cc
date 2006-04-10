//CMSSW headers
#include "Geometry/Surface/interface/BoundDisk.h"
#include "Geometry/Surface/interface/BoundCylinder.h"

//FAMOS headers
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"

#include <iomanip>

ParticlePropagator::ParticlePropagator(const RawParticle& myPart,
				       double R, double Z, double B) :
  BaseParticlePropagator(myPart,R,Z,B) 
{
  setMagneticField(fieldMap(x(),y(),z()));
}

ParticlePropagator::ParticlePropagator( const RawParticle& myPart ) :
  BaseParticlePropagator(myPart,0.,0.,0.) 
{
  setMagneticField(fieldMap(x(),y(),z()));
}

ParticlePropagator::ParticlePropagator(const HepLorentzVector& mom, 
				       const HepLorentzVector& vert, float q) :
  BaseParticlePropagator(RawParticle(mom,vert),0.,0.,0.) 
{
  setCharge(q);
  setMagneticField(fieldMap(x(),y(),z()));
}

ParticlePropagator::ParticlePropagator(const HepLorentzVector& mom, 
				       const Hep3Vector& vert, float q) :
  BaseParticlePropagator(RawParticle(mom,HepLorentzVector(vert,0.0)),0.,0.,0.) 
{
  setCharge(q);
  setMagneticField(fieldMap(x(),y(),z()));
}

ParticlePropagator::ParticlePropagator(const FSimTrack& simTrack) : 
  BaseParticlePropagator(RawParticle(simTrack.type(),simTrack.momentum()),
			 0.,0.,0.)
{
  setVertex(simTrack.vertex().position());
  setMagneticField(fieldMap(x(),y(),z()));
}

ParticlePropagator::ParticlePropagator(const BaseParticlePropagator& myPropPart) :
  BaseParticlePropagator(myPropPart)
{  
  setMagneticField(fieldMap(x(),y(),z()));
}

bool
ParticlePropagator::propagateToClosestApproach(bool first) {
  setMagneticField(fieldMap(0.,0.,0.));
  return BaseParticlePropagator::propagateToClosestApproach(first);
}

bool
ParticlePropagator::propagateToNominalVertex(const HepLorentzVector& v) {
  setMagneticField(fieldMap(0.,0.,0.));
  return BaseParticlePropagator::propagateToNominalVertex(v);
}

ParticlePropagator
ParticlePropagator::propagated() const {
  return ParticlePropagator(BaseParticlePropagator::propagated());
}

double
ParticlePropagator::fieldMap(double x,double y, double z) {
  return BaseParticlePropagator::getMagneticField();
}

bool
ParticlePropagator::propagateToBoundSurface(const TrackerLayer& layer) {


  fiducial = true;
  bool disk = layer.forward();
  double innerradius=-999;

  if( disk ) {
    const Surface& surface = layer.surface();
    const BoundDisk & myDisk = dynamic_cast<const BoundDisk&>(surface);
    innerradius=myDisk.innerRadius()*10.;	  
  }

  bool done = propagate();

  // There is some real material here
  fiducial = !(!disk &&  success!=1) &&
	     !( disk && (success!=2  || position().perp()<innerradius));

  return done;
}

void 
ParticlePropagator::setPropagationConditions(const TrackerLayer& layer, 
					     bool firstLoop) { 

  const Surface& surface = layer.surface();

  if( layer.forward() ) {

    const BoundDisk & myDisk = dynamic_cast<const BoundDisk&>(surface);
    // ParticlePropagator works in mm, whereas the detector geometry is in cm
    BaseParticlePropagator::setPropagationConditions(
                                  myDisk.outerRadius()*10.,
				  fabs(myDisk.position().z())*10.,
				  firstLoop);       

  // ... or if it is a cylinder barrel 
  } else {

    const BoundCylinder & myCylinder = dynamic_cast<const BoundCylinder &>(surface);
    // ParticlePropagator works in mm, whereas the detector geometry is in cm
    BaseParticlePropagator::setPropagationConditions(
					 myCylinder.radius()*10.,
					 myCylinder.bounds().length()/2.*10.,
					 firstLoop);
  }

}


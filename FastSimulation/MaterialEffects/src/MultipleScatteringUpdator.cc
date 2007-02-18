#include "FastSimulation/MaterialEffects/interface/MultipleScatteringUpdator.h"

#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace std;

MultipleScatteringUpdator::MultipleScatteringUpdator(
  const RandomEngine* engine) :
    MaterialEffectsUpdator(engine)
{
  ;
}

void MultipleScatteringUpdator::compute(ParticlePropagator &Particle)
{

  double p    = Particle.vect().mag();
  double mass = Particle.mass();
  double e    = sqrt(p*p+mass*mass);

  double beta = p/e;

  // Average multiple scattering angle from Moliere radius
  double theta0 = 0.0136 / (p*beta) * Particle.charge() 
                                    * sqrt(radLengths) 
                                    * (1. + 0.038*log(radLengths));

  // Generate multiple scattering angles in the two directions 
  // perpendicular to the particle motion
  double theta1 = random->gaussShoot(0.,theta0); 
  double theta2 = random->gaussShoot(0.,theta0); 

  HepVector3D theP(Particle.px(),Particle.py(),Particle.pz());
  HepVector3D perpP1 = theP.orthogonal();
  HepVector3D perpP2 = theP.cross(perpP1);
  HepTransform3D Rotation1 = HepRotate3D(theta1,perpP1);
  HepTransform3D Rotation2 = HepRotate3D(theta2,perpP2);
  HepTransform3D theRotation = Rotation2*Rotation1;
  HepVector3D newP(theRotation*theP);

  Particle.setPx(newP.x()); 
  Particle.setPy(newP.y()); 
  Particle.setPz(newP.z());

  // Generate mutiple scattering displacements in mm (assuming the detectors
  // are silicon only to determine the thickness) in the directions orthogonal
  // to the vector normal to the surface

  double xp = (theta1/2. + random->gaussShoot(0.,theta0)/sqrt(12.))
                         * radLengths * radLenIncm() * 10.;
  double yp = (theta2/2. + random->gaussShoot(0.,theta0)/sqrt(12.))
                         * radLengths * radLenIncm() * 10.;
  
  HepVector3D 
    normal(theNormalVector.x(),theNormalVector.y(),theNormalVector.z());
  HepVector3D tangent1 = normal.orthogonal();
  HepVector3D tangent2 = normal.cross(tangent1);
  HepVector3D thePos(Particle.vertex().x(),
		     Particle.vertex().y(),
		     Particle.vertex().z());

  HepVector3D newPos = thePos + tangent1*xp + tangent2*yp;

  HepLorentzVector newVertex(newPos,Particle.vertex().t());

  Particle.setVertex(newVertex);

}


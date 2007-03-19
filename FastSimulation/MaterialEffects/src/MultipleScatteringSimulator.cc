#include "FastSimulation/MaterialEffects/interface/MultipleScatteringSimulator.h"

#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace std;

MultipleScatteringSimulator::MultipleScatteringSimulator(
  const RandomEngine* engine) :
    MaterialEffectsSimulator(engine)
{
  ;
}

void MultipleScatteringSimulator::compute(ParticlePropagator &Particle)
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

  Hep3Vector axis1 = Particle.vect().orthogonal();
  Hep3Vector axis2 = Particle.vect().cross(axis1);
  HepRotation theRotation1(axis1,theta1);
  HepRotation theRotation2(axis2,theta2);
  Particle *= theRotation2*theRotation1;

  // Generate mutiple scattering displacements in mm (assuming the detectors
  // are silicon only to determine the thickness) in the directions orthogonal
  // to the vector normal to the surface
  double xp = (theta1/2. + random->gaussShoot(0.,theta0)/sqrt(12.))
                         * radLengths * radLenIncm();		 
  double yp = (theta2/2. + random->gaussShoot(0.,theta0)/sqrt(12.))
                         * radLengths * radLenIncm();
  
  Hep3Vector 
    normal(theNormalVector.x(),theNormalVector.y(),theNormalVector.z());
  Hep3Vector tangent1 = normal.orthogonal();
  Hep3Vector tangent2 = normal.cross(tangent1);

  Hep3Vector newPos = Particle.vertex().vect() + tangent1*xp + tangent2*yp;

  HepLorentzVector newVertex(newPos,Particle.vertex().t());

  Particle.setVertex(newVertex);

}


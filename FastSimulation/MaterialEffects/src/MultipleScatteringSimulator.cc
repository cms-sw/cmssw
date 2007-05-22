#include "FastSimulation/MaterialEffects/interface/MultipleScatteringSimulator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include <iostream>
#include <cmath>

MultipleScatteringSimulator::MultipleScatteringSimulator(
  const RandomEngine* engine) :
    MaterialEffectsSimulator(engine)
{
  sqr12 = std::sqrt(12.);
}

void MultipleScatteringSimulator::compute(ParticlePropagator &Particle)
{

  double p2 = Particle.Vect().Mag2();
  double m2 = Particle.mass()*Particle.mass();
  double e = std::sqrt(p2+m2);

  double pbeta = p2/e;  // This is p*beta

  // Average multiple scattering angle from Moliere radius
  double theta0 = 0.0136 / pbeta * Particle.charge() 
                                 * std::sqrt(radLengths) 
                                 * (1. + 0.038*std::log(radLengths));

  // Generate multiple scattering angles in the two directions 
  // perpendicular to the particle motion
  double theta1 = random->gaussShoot(0.,theta0); 
  double theta2 = random->gaussShoot(0.,theta0); 

  XYZVector axis1 = orthogonal(Particle.Vect());
  XYZVector axis2 = Particle.Vect().Cross(axis1);
  RawParticle::Rotation rotation1(axis1,theta1);
  RawParticle::Rotation rotation2(axis2,theta2);
  Particle.rotate(rotation1); 
  Particle.rotate(rotation2);

  // Generate mutiple scattering displacements in mm (assuming the detectors
  // are silicon only to determine the thickness) in the directions orthogonal
  // to the vector normal to the surface
  double xp = (theta1/2. + random->gaussShoot(0.,theta0)/sqr12)
                         * radLengths * radLenIncm();		 
  double yp = (theta2/2. + random->gaussShoot(0.,theta0)/sqr12)
                         * radLengths * radLenIncm();

  XYZVector normal(theNormalVector.x(),theNormalVector.y(),theNormalVector.z());
  XYZVector tangent = orthogonal(normal); // This vector is unitary because 
                                          // normal is
                                          // either (0,0,1) in the Endcap 
                                          // or     (x,y,0) in the Barrel !
  XYZVector delta = xp*tangent + yp*normal.Cross(tangent);

  Particle.translate(delta);

}


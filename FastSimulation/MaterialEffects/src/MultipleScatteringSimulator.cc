#include "FastSimulation/MaterialEffects/interface/MultipleScatteringSimulator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include <cmath>

MultipleScatteringSimulator::MultipleScatteringSimulator(
  const RandomEngine* engine, double A, double Z, double density, double radLen) :
    MaterialEffectsSimulator(engine,A,Z,density,radLen)
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
  // The sqrt(2) factor is because of the *space* angle
  double theta0 = 0.0136 / pbeta * Particle.charge() 
                                 * std::sqrt(2.*radLengths) 
                                 * (1. + 0.038*std::log(radLengths));

  // Generate multiple scattering space angle perpendicular to the particle motion
  double theta = random->gaussShoot(0.,theta0); 
  // Plus a random rotation angle around the particle motion
  double phi = 2. * 3.14159265358979323 * random->flatShoot();
  // The two rotations
  RawParticle::Rotation rotation1(orthogonal(Particle.Vect()),theta);
  RawParticle::Rotation rotation2(Particle.Vect(),phi);
  // Rotate!
  Particle.rotate(rotation1); 
  Particle.rotate(rotation2);

  // Generate mutiple scattering displacements in cm (assuming the detectors
  // are silicon only to determine the thickness) in the directions orthogonal
  // to the vector normal to the surface
  double xp = (cos(phi)*theta/2. + random->gaussShoot(0.,theta0)/sqr12)
              * radLengths * radLenIncm();		 
  double yp = (sin(phi)*theta/2. + random->gaussShoot(0.,theta0)/sqr12)
              * radLengths * radLenIncm();

  // Determine a unitary vector tangent to the surface
  // This tangent vector is unitary because "normal" is
  // either (0,0,1) in the Endcap  or (x,y,0) in the Barrel !
  XYZVector normal(theNormalVector.x(),theNormalVector.y(),theNormalVector.z());
  XYZVector tangent = orthogonal(normal); 
  // The total displacement 
  XYZVector delta = xp*tangent + yp*normal.Cross(tangent);
  // Translate!
  Particle.translate(delta);

}


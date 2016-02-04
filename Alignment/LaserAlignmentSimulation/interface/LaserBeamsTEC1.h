#ifndef LaserAlignmentSimulation_LaserBeamsTEC1_h
#define LaserAlignmentSimulation_LaserBeamsTEC1_h

/** \class LaserBeamsTEC1
 *  Laser Beams in the right Endcap
 *
 *  $Date: 2009/05/26 07:12:23 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

#include "CLHEP/Random/DRand48Engine.h"

// G4 includes
#include "G4ParticleTable.hh"
#include "G4VUserPrimaryGeneratorAction.hh"

class G4ParticleGun;
class G4Event;

class LaserBeamsTEC1 : public G4VUserPrimaryGeneratorAction
{
 public:
	/// default constructor
  LaserBeamsTEC1();
	/// constructor
  LaserBeamsTEC1(G4int nPhotonsInGun, G4int nPhotonsInBeam, G4double PhotonEnergy);
	/// destructor
  ~LaserBeamsTEC1();

 public:
	/// shoot optical photons into the detector at the beginning of an event
	void GeneratePrimaries(G4Event* myEvent);
	/// set the polarisation of the photons
  void setOptPhotonPolar(G4double Angle);

 private:
  G4int thenParticleInGun;
  G4int thenParticle;
  G4double thePhotonEnergy;

 private:
  G4ParticleGun * theParticleGun;

  // Unique random number generator
  CLHEP::DRand48Engine* theDRand48Engine;
};
#endif


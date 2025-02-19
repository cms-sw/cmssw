#ifndef LaserAlignmentSimulation_LaserBeamsBarrel_h
#define LaserAlignmentSimulation_LaserBeamsBarrel_h

/** \class LaserBeamsBarrel
 *  Define the LaserBeams which connect both TECs and TIB and TOB with eachother
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

class LaserBeamsBarrel : public G4VUserPrimaryGeneratorAction
{
 public:
	/// default constructor
  LaserBeamsBarrel();
	/// constructor
  LaserBeamsBarrel(G4int nPhotonsInGun, G4int nPhotonsInBeam, G4double PhotonEnergy);
	/// destructor
  ~LaserBeamsBarrel();

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


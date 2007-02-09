/* 
 * Define the LaserBeams which
 * connect both TECs and TIB
 * and TOB with eachother
 */
#ifndef LaserAlignmentSimulation_LaserBeamsBarrel_h
#define LaserAlignmentSimulation_LaserBeamsBarrel_h


#include "CLHEP/Random/DRand48Engine.h"
#include "CLHEP/Random/RandGaussQ.h"

// G4 includes
#include "globals.hh"                        // Global Constants and typedefs
#include "G4DataVector.hh"
#include "G4Event.hh"
#include "G4OpticalPhoton.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4VUserPrimaryGeneratorAction.hh"

class G4ParticleGun;
class G4Event;

class LaserBeamsBarrel : public G4VUserPrimaryGeneratorAction
{
 public:
  LaserBeamsBarrel();
  LaserBeamsBarrel(G4int nPhotonsInGun, G4int nPhotonsInBeam, G4double PhotonEnergy);
  ~LaserBeamsBarrel();

 public:
  void GeneratePrimaries(G4Event* myEvent);
  void setOptPhotonPolar(G4double Angle);

 private:
  G4int thenParticleInGun;
  G4int thenParticle;
  G4double thePhotonEnergy;

 private:
  G4ParticleGun * theParticleGun;

  // Unique random number generator
  DRand48Engine* theDRand48Engine;
};
#endif


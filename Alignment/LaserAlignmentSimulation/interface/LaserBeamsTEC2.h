#ifndef LaserAlignmentSimulation_LaserBeamsTEC2_h
#define LaserAlignmentSimulation_LaserBeamsTEC2_h

/** \class LaserBeamsTEC2
 *  Laser Beams in the left Endcap
 *
 *  $Date: Mon Mar 19 12:04:13 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

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

class LaserBeamsTEC2 : public G4VUserPrimaryGeneratorAction
{
 public:
	/// default constructor
  LaserBeamsTEC2();
	/// constructor
  LaserBeamsTEC2(G4int nPhotonsInGun, G4int nPhotonsInBeam, G4double PhotonEnergy);
	/// destructor
  ~LaserBeamsTEC2();

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
  DRand48Engine* theDRand48Engine;
};
#endif


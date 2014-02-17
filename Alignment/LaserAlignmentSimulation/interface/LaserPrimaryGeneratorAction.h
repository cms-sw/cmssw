#ifndef LaserAlignmentSimulation_LaserPrimaryGeneratorAction_h
#define LaserAlignmentSimulation_LaserPrimaryGeneratorAction_h

/** \class LaserPrimaryGeneratorAction
 *  Primary Generator Action for the Laser Events
 *
 *  $Date: 2007/06/11 14:44:28 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/interface/LaserBeamsTEC1.h"
#include "Alignment/LaserAlignmentSimulation/interface/LaserBeamsTEC2.h"
#include "Alignment/LaserAlignmentSimulation/interface/LaserBeamsBarrel.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// G4 includes
#include "G4Event.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

class G4Event;
class LaserPrimaryGeneratorMessenger;

class LaserPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
 public:
	/// constructor
  LaserPrimaryGeneratorAction(edm::ParameterSet const& theConf);
	/// destructor
  ~LaserPrimaryGeneratorAction();

 public:
	/// call the corresponding GeneratePrimaries routines for both TEC's and the Barrel
  void GeneratePrimaries(G4Event* myEvent);
	/// set Id of the optical photons
  void setGeneratorId(G4PrimaryParticle * aParticle, int ID) const;

 private:
  G4double thePhotonEnergy;
  G4int thenParticleInGun;
  G4int thenParticle;

 private:
  LaserBeamsTEC1 * theLaserBeamsInTEC1;
  LaserBeamsTEC2 * theLaserBeamsInTEC2;
  LaserBeamsBarrel * theLaserBeamsInTECTIBTOBTEC;
};
#endif

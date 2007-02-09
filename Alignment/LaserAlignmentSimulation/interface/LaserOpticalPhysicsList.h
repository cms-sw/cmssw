/* 
 * Define the Optical processes for the Simulation of 
 * the Laser Alignment System with CMSSW
 */

#ifndef LaserAlignmentSimulation_LaserOpticalPhysicsList_H
#define LaserAlignmentSimulation_LaserOpticalPhysicsList_H

// G4 includes
#include "globals.hh"
#include "G4VPhysicsConstructor.hh"

#include "G4ios.hh"
#include <iomanip>

#include "G4Material.hh"
#include "G4MaterialTable.hh"

#include "G4ParticleDefinition.hh"
#include "G4ParticleTypes.hh"
#include "G4ParticleTable.hh"

#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"

#include "G4Cerenkov.hh"
#include "G4Scintillation.hh"
#include "G4OpAbsorption.hh"
#include "G4OpRayleigh.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4OpWLS.hh"

class G4Cerenkov;
class G4Scintillation;
class G4OpAbsorption;
class G4OpRayleigh;
class G4OpBoundaryProcess;

class LaserOpticalPhysicsList : public G4VPhysicsConstructor
{
 public:
  LaserOpticalPhysicsList(const G4String& name="optical");
  virtual  ~LaserOpticalPhysicsList();

 public:
  virtual void ConstructParticle();
  virtual void ConstructProcess();

 protected:
  G4bool wasActivated;

  G4Scintillation* theScintProcess;
  G4Cerenkov* theCerenkovProcess;
  G4OpAbsorption* theAbsorptionProcess;
  G4OpRayleigh* theRayleighScattering;
  G4OpBoundaryProcess* theBoundaryProcess;
  G4OpWLS* theWLSProcess;
};
#endif

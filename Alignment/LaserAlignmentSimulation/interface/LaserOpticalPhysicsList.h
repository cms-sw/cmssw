#ifndef LaserAlignmentSimulation_LaserOpticalPhysicsList_H
#define LaserAlignmentSimulation_LaserOpticalPhysicsList_H

/** \class LaserOpticalPhysicsList
 *  Define the Optical processes for the Simulation of the Laser Alignment System
 *
 *  $Date: Mon Mar 19 12:04:31 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

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
	/// constructor
  LaserOpticalPhysicsList(const G4String& name="optical");
	/// destructor
  virtual  ~LaserOpticalPhysicsList();

 public:
	/// construct Optical Photons
  virtual void ConstructParticle();
	/// construct Optical Processes
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

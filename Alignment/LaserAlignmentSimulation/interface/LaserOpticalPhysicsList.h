#ifndef LaserAlignmentSimulation_LaserOpticalPhysicsList_H
#define LaserAlignmentSimulation_LaserOpticalPhysicsList_H

/** \class LaserOpticalPhysicsList
 *  Define the Optical processes for the Simulation of the Laser Alignment System
 *
 *  $Date: 2007/12/04 23:53:05 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

// G4 includes
#include "G4VPhysicsConstructor.hh"

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

/** \file LaserOpticalPhysicsList.cc
 *  
 *
 *  $Date: 2008/03/10 12:52:52 $
 *  $Revision: 1.5 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/interface/LaserOpticalPhysicsList.h"
#include "G4ProcessManager.hh" 

#include "G4Cerenkov.hh" 
#include "G4Scintillation.hh" 
#include "G4OpAbsorption.hh" 
#include "G4OpRayleigh.hh" 
#include "G4OpBoundaryProcess.hh" 

LaserOpticalPhysicsList::LaserOpticalPhysicsList(const G4String& name)
  :  G4VPhysicsConstructor(name),
     wasActivated(false),
     theScintProcess(), 
     theCerenkovProcess(), 
     theAbsorptionProcess(),
     theRayleighScattering(), 
     theBoundaryProcess(), 
     theWLSProcess()
{
  if (verboseLevel > 0)
    std::cout << "<LaserOpticalPhysicsList::LaserOpticalPhysicsList(...)> entering constructor ..." << std::endl;
}

LaserOpticalPhysicsList::~LaserOpticalPhysicsList()
{
  if (verboseLevel > 0)
    {
      std::cout << "<LaserOpticalPhysicsList::~LaserOpticalPhysicsList()> entering destructor ... " << std::endl;
      std::cout << "  deleting the processes ... ";
    }
  if ( theWLSProcess != 0 )         { delete theWLSProcess; }
  if ( theBoundaryProcess != 0 )    { delete theBoundaryProcess; }
  if ( theRayleighScattering != 0 ) { delete theRayleighScattering; }
  if ( theAbsorptionProcess != 0 )  { delete theAbsorptionProcess; }
//  if ( theCerenkovProcess != 0 )    { delete theCerenkovProcess; }
  if ( theScintProcess != 0 )       { delete theScintProcess; }
  if (verboseLevel > 0)
    std::cout << " done " << std::endl;
}

void LaserOpticalPhysicsList::ConstructParticle()
{
  if (verboseLevel > 0)
    std::cout << "<LaserOpticalPhysicsList::ConstructParticle()>: constructing the optical photon ... " << std::endl;

  // optical photon
  G4OpticalPhoton::OpticalPhotonDefinition();
}

void LaserOpticalPhysicsList::ConstructProcess()
{
  if (verboseLevel > 0)
    std::cout << "<LaserOpticalPhysicsList::ConstructProcess()>: constructing the physics ... " << std::endl;

  theScintProcess = new G4Scintillation();
//  theCerenkovProcess=new G4Cerenkov();
  theAbsorptionProcess=new G4OpAbsorption();
  theRayleighScattering=new G4OpRayleigh();
  theBoundaryProcess=new G4OpBoundaryProcess("OpBoundary");
  theWLSProcess=new G4OpWLS();
  
  // set the verbosity level
  theAbsorptionProcess->SetVerboseLevel(verboseLevel);
  theBoundaryProcess->SetVerboseLevel(verboseLevel);

  G4ProcessManager * pManager = 0;
  
  pManager = G4OpticalPhoton::OpticalPhoton()->GetProcessManager();
  pManager->AddDiscreteProcess(theAbsorptionProcess);
  pManager->AddDiscreteProcess(theRayleighScattering);
  //theBoundaryProcess->SetModel(unified);
  pManager->AddDiscreteProcess(theBoundaryProcess);
  pManager->AddDiscreteProcess(theWLSProcess);
  
  theScintProcess->SetScintillationYieldFactor(1.);
  theScintProcess->SetScintillationExcitationRatio(0.0);
  theScintProcess->SetTrackSecondariesFirst(true);
  
  aParticleIterator->reset();
  while( (*aParticleIterator)() )
    {
      G4ParticleDefinition* particle = aParticleIterator->value();
      pManager = particle->GetProcessManager();
//      if(theCerenkovProcess->IsApplicable(*particle))
//	{
//	  pManager->AddContinuousProcess(theCerenkovProcess);
//	}
      if(theScintProcess->IsApplicable(*particle))
	{
	  pManager->AddProcess(theScintProcess);
	  pManager->SetProcessOrderingToLast(theScintProcess,idxAtRest);
	  pManager->SetProcessOrderingToLast(theScintProcess,idxPostStep);
	}
    }

  wasActivated = true;
}

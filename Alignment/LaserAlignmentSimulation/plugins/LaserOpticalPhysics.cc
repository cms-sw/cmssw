/*
 * Custom Physics to activate
 * optical processes for the 
 * simulation of the Laser Alignment System
 */

#include "Alignment/LaserAlignmentSimulation/plugins/LaserOpticalPhysics.h"
#include "Alignment/LaserAlignmentSimulation/interface/LaserOpticalPhysicsList.h"
 
#include "SimG4Core/QGSP/src/HadronPhysicsQGSP.hh"

#include "SimG4Core/Physics/interface/PhysicsListFactory.h" 

#ifdef G4V7
#include "SimG4Core/Packaging/src/GeneralPhysics.hh"
#include "SimG4Core/Packaging/src/EMPhysics.hh"
#include "SimG4Core/Packaging/src/MuonPhysics.hh"
#include "SimG4Core/Packaging/src/IonPhysics.hh"
#include "SimG4Core/Packaging/src/G4DataQuestionaire.hh"
#else
#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 
#include "G4DataQuestionaire.hh"
#endif

LaserOpticalPhysics::LaserOpticalPhysics(const edm::ParameterSet & p) : PhysicsList(p)
{
    G4DataQuestionaire it(photon);
#ifdef G4V7
    std::cout << "You are using the simulation engine: QGSP 2.8" << std::endl;
  
    RegisterPhysics(new GeneralPhysics("general"));
    RegisterPhysics(new EMPhysics("EM"));
    RegisterPhysics(new MuonPhysics("muon"));
    RegisterPhysics(new HadronPhysicsQGSP("hadron"));
    RegisterPhysics(new IonPhysics("ion"));
		// Optical physics
		RegisterPhysics(new LaserOpticalPhysicsList("optical"));
#else
    std::cout << "You are using the simulation engine: QGSP 3.1" << std::endl;
  
    // EM Physics
    RegisterPhysics(new G4EmStandardPhysics("standard EM"));
    // Synchroton Radiation & GN Physics
    RegisterPhysics(new G4EmExtraPhysics("extra EM"));
    // Decays
    RegisterPhysics(new G4DecayPhysics("decay"));
    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysics("elastic")); 
    // Hadron Physics
    RegisterPhysics(new HadronPhysicsQGSP("hadron"));
    // Stopping Physics
    RegisterPhysics(new G4QStoppingPhysics("stopping"));
    // Ion Physics
    RegisterPhysics(new G4IonPhysics("ion"));
		// Optical physics
		RegisterPhysics(new LaserOpticalPhysicsList("optical"));
#endif

}

// define the custom physics list
#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE ();
DEFINE_PHYSICSLIST (LaserOpticalPhysics);

/** \file LaserOpticalPhysics.cc
 *  Custom Physics to activate optical processes for the simulation of the Laser Alignment System
 *
 *  $Date: 2007/08/27 19:46:11 $
 *  $Revision: 1.6 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/plugins/LaserOpticalPhysics.h"
#include "Alignment/LaserAlignmentSimulation/interface/LaserOpticalPhysicsList.h"
 
#include "HadronPhysicsQGSP.hh"

#include "SimG4Core/Physics/interface/PhysicsListFactory.h" 

#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 
#include "G4DataQuestionaire.hh"

LaserOpticalPhysics::LaserOpticalPhysics(G4LogicalVolumeToDDLogicalPartMap& map,
  const edm::ParameterSet & p) : PhysicsList(map, p)
{
    G4DataQuestionaire it(photon);
    std::cout << "You are using the simulation engine: QGSP together with optical physics" << std::endl;
  
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

}

// define the custom physics list
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE ();
DEFINE_PHYSICSLIST (LaserOpticalPhysics);

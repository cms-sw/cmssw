/** \file LaserOpticalPhysics.cc
 *  Custom Physics to activate optical processes for the simulation of the Laser Alignment System
 *
 *  $Date: 2010/08/02 13:09:56 $
 *  $Revision: 1.11 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/plugins/LaserOpticalPhysics.h"
#include "Alignment/LaserAlignmentSimulation/interface/LaserOpticalPhysicsList.h"
 
#include "HadronPhysicsQGSP.hh"

#include "SimG4Core/Physics/interface/PhysicsListFactory.h" 
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 
#include "G4DataQuestionaire.hh"

LaserOpticalPhysics::LaserOpticalPhysics(G4LogicalVolumeToDDLogicalPartMap& map,
					 const HepPDT::ParticleDataTable * table_,
					 sim::FieldBuilder *fieldBuilder_,
					 const edm::ParameterSet & p) : PhysicsList(map, table_, fieldBuilder_, p)
{
    G4DataQuestionaire it(photon);
    std::cout << "You are using the simulation engine: QGSP together with optical physics" << std::endl;
  
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics("standard EM", 0));
    // Synchroton Radiation & GN Physics
    RegisterPhysics(new G4EmExtraPhysics("extra EM"));
    // Decays
    RegisterPhysics(new G4DecayPhysics("decay",0));
    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysics("elastic",0,false)); 
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

DEFINE_PHYSICSLIST (LaserOpticalPhysics);

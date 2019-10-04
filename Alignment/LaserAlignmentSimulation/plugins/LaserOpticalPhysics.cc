/** \file LaserOpticalPhysics.cc
 *  Custom Physics to activate optical processes for the simulation of the Laser
 * Alignment System
 *
 *  $Date: 2010/08/02 13:09:56 $
 *  $Revision: 1.11 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/interface/LaserOpticalPhysicsList.h"
#include "Alignment/LaserAlignmentSimulation/plugins/LaserOpticalPhysics.h"

#include "G4HadronPhysicsQGSP_FTFP_BERT.hh"

#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4HadronicProcessStore.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"

LaserOpticalPhysics::LaserOpticalPhysics(const edm::ParameterSet &p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  std::cout << "You are using the simulation engine: QGSP together with "
               "optical physics"
            << std::endl;

  // EM Physics
  RegisterPhysics(new CMSEmStandardPhysics(ver));
  // Synchroton Radiation & GN Physics
  RegisterPhysics(new G4EmExtraPhysics(ver));
  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));
  // Hadron Elastic scattering
  G4HadronicProcessStore::Instance()->SetVerbose(ver);
  RegisterPhysics(new G4HadronElasticPhysics(ver));
  // Hadron Physics
  RegisterPhysics(new G4HadronPhysicsQGSP_FTFP_BERT(ver));
  // Stopping Physics
  RegisterPhysics(new G4StoppingPhysics(ver));
  // Ion Physics
  RegisterPhysics(new G4IonPhysics(ver));
  // Optical physics
  RegisterPhysics(new LaserOpticalPhysicsList("optical"));
}

// define the custom physics list

DEFINE_PHYSICSLIST(LaserOpticalPhysics);

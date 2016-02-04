#ifndef LaserAlignmentSimulation_LaserOpticalPhysics_H
#define LaserAlignmentSimulation_LaserOpticalPhysics_H

/** \class LaserOpticalPhysics
 *  Custom physics to activate optical processes for the simulation of the Laser Alignment System
 *
 *  $Date: 2010/08/02 13:09:56 $
 *  $Revision: 1.5 $
 *  \author Maarten Thomas
 */ 

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class LaserOpticalPhysics : public PhysicsList
{
public:
		/// constructor
    LaserOpticalPhysics(G4LogicalVolumeToDDLogicalPartMap& map, const HepPDT::ParticleDataTable * table_, sim::FieldBuilder *fieldBuilder_, const edm::ParameterSet & p);
};
 
#endif

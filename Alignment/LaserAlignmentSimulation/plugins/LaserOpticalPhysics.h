#ifndef LaserAlignmentSimulation_LaserOpticalPhysics_H
#define LaserAlignmentSimulation_LaserOpticalPhysics_H

/** \class LaserOpticalPhysics
 *  Custom physics to activate optical processes for the simulation of the Laser Alignment System
 *
 *  $Date: 2007/03/20 12:01:00 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */ 

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class LaserOpticalPhysics : public PhysicsList
{
public:
		/// constructor
    LaserOpticalPhysics(G4LogicalVolumeToDDLogicalPartMap& map, const edm::ParameterSet & p);
};
 
#endif

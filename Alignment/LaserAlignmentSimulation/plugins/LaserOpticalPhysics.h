#ifndef LaserAlignmentSimulation_LaserOpticalPhysics_H
#define LaserAlignmentSimulation_LaserOpticalPhysics_H

/** \class LaserOpticalPhysics
 *  Custom physics to activate optical processes for the simulation of the Laser Alignment System
 *
 *  $Date: Mon Mar 19 12:27:44 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */ 

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class LaserOpticalPhysics : public PhysicsList
{
public:
		/// constructor
    LaserOpticalPhysics(const edm::ParameterSet & p);
};
 
#endif

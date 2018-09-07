#ifndef LaserAlignmentSimulation_LaserOpticalPhysics_H
#define LaserAlignmentSimulation_LaserOpticalPhysics_H

/** \class LaserOpticalPhysics
 *  Custom physics to activate optical processes for the simulation of the Laser Alignment System
 *
 *  $Date: 2009/12/10 18:10:05 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */ 

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class LaserOpticalPhysics : public PhysicsList
{
public:

  LaserOpticalPhysics(const edm::ParameterSet & p);
};
 
#endif

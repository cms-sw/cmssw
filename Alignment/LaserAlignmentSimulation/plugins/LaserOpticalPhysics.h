/* 
 * Custom physics to activate
 * optical processes for the simulation
 */

#ifndef LaserAlignmentSimulation_LaserOpticalPhysics_H
#define LaserAlignmentSimulation_LaserOpticalPhysics_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class LaserOpticalPhysics : public PhysicsList
{
public:
    LaserOpticalPhysics(const edm::ParameterSet & p);
};
 
#endif

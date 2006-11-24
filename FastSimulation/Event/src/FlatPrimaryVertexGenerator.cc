//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Famos Headers
#include "FastSimulation/Event/interface/FlatPrimaryVertexGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

  /// Default constructor
FlatPrimaryVertexGenerator::FlatPrimaryVertexGenerator(const edm::ParameterSet& vtx) : 
  PrimaryVertexGenerator(),
  minX(vtx.getParameter<double>("MinX")),
  minY(vtx.getParameter<double>("MinY")),
  minZ(vtx.getParameter<double>("MinZ")),
  maxX(vtx.getParameter<double>("MaxX")),
  maxY(vtx.getParameter<double>("MaxY")),
  maxZ(vtx.getParameter<double>("MaxZ"))
{}
  
void
FlatPrimaryVertexGenerator::generate() {

  this->setX(random->flatShoot(minX,maxX));
  this->setY(random->flatShoot(minY,maxY));
  this->setZ(random->flatShoot(minZ,maxZ));

}

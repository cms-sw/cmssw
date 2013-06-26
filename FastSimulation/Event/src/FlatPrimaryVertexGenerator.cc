//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Famos Headers
#include "FastSimulation/Event/interface/FlatPrimaryVertexGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

  /// Default constructor
FlatPrimaryVertexGenerator::FlatPrimaryVertexGenerator(
  const edm::ParameterSet& vtx, const RandomEngine* engine) : 
  PrimaryVertexGenerator(engine),
  minX(vtx.getParameter<double>("MinX")),
  minY(vtx.getParameter<double>("MinY")),
  minZ(vtx.getParameter<double>("MinZ")),
  maxX(vtx.getParameter<double>("MaxX")),
  maxY(vtx.getParameter<double>("MaxY")),
  maxZ(vtx.getParameter<double>("MaxZ"))
{
  beamSpot_ = math::XYZPoint((minX+maxX)/2.,(minY+maxY)/2.,(minZ+maxZ)/2.);
}
  
void
FlatPrimaryVertexGenerator::generate() {

  this->SetX(random->flatShoot(minX,maxX));
  this->SetY(random->flatShoot(minY,maxY));
  this->SetZ(random->flatShoot(minZ,maxZ));

}

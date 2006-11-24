//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Famos Headers
#include "FastSimulation/Event/interface/GaussianPrimaryVertexGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

  /// Default constructor
GaussianPrimaryVertexGenerator::GaussianPrimaryVertexGenerator(const edm::ParameterSet& vtx) : 
  PrimaryVertexGenerator(),
  meanX(vtx.getParameter<double>("MeanX")),
  meanY(vtx.getParameter<double>("MeanY")),
  meanZ(vtx.getParameter<double>("MeanZ")),
  sigmaX(vtx.getParameter<double>("SigmaX")),
  sigmaY(vtx.getParameter<double>("SigmaY")),
  sigmaZ(vtx.getParameter<double>("SigmaZ"))
{}
  
void
GaussianPrimaryVertexGenerator::generate() {

  this->setX(random->gaussShoot(meanX,sigmaX));
  this->setY(random->gaussShoot(meanY,sigmaY));
  this->setZ(random->gaussShoot(meanZ,sigmaZ));

}

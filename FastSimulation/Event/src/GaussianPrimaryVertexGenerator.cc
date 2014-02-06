//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Famos Headers
#include "FastSimulation/Event/interface/GaussianPrimaryVertexGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

  /// Default constructor
GaussianPrimaryVertexGenerator::GaussianPrimaryVertexGenerator(
  const edm::ParameterSet& vtx) :
  meanX(vtx.getParameter<double>("MeanX")),
  meanY(vtx.getParameter<double>("MeanY")),
  meanZ(vtx.getParameter<double>("MeanZ")),
  sigmaX(vtx.getParameter<double>("SigmaX")),
  sigmaY(vtx.getParameter<double>("SigmaY")),
  sigmaZ(vtx.getParameter<double>("SigmaZ"))
{
  beamSpot_ = math::XYZPoint(meanX,meanY,meanZ);
}
  
void
GaussianPrimaryVertexGenerator::generate(RandomEngineAndDistribution const* random) {

  this->SetX(random->gaussShoot(meanX,sigmaX));
  this->SetY(random->gaussShoot(meanY,sigmaY));
  this->SetZ(random->gaussShoot(meanZ,sigmaZ));

}

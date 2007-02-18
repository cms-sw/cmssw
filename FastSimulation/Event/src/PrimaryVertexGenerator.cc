#include "FastSimulation/Event/interface/PrimaryVertexGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

  /// Default constructor
PrimaryVertexGenerator::PrimaryVertexGenerator() : 
  Hep3Vector(), 
  random(0)  
{
}

PrimaryVertexGenerator::PrimaryVertexGenerator(const RandomEngine* engine) : 
  Hep3Vector(), 
  random(engine)  
{
}
  

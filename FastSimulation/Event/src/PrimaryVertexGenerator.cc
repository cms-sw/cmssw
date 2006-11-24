#include "FastSimulation/Event/interface/PrimaryVertexGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

  /// Default constructor
PrimaryVertexGenerator::PrimaryVertexGenerator() : 
  Hep3Vector(), 
  random(RandomEngine::instance())  
{
}
  

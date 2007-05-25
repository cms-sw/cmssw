#include "FastSimulation/Event/interface/PrimaryVertexGenerator.h"

  /// Default constructor
PrimaryVertexGenerator::PrimaryVertexGenerator() : 
  math::XYZVector(), 
  random(0)  
{
}

PrimaryVertexGenerator::PrimaryVertexGenerator(const RandomEngine* engine) : 
  math::XYZVector(), 
  random(engine)  
{
}
  

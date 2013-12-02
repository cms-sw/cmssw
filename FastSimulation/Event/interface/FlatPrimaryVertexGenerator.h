#ifndef FastSimulation_Event_FlatPrimaryVertexGenerator_H
#define FastSimulation_Event_FlatPrimaryVertexGenerator_H

// Famos Headers
#include "FastSimulation/Event/interface/PrimaryVertexGenerator.h"

/** A class that generates a primary vertex with a Gaussian profile, in cm*/ 

class RandomEngineAndDistribution;

namespace edm { 
  class ParameterSet;
}

class FlatPrimaryVertexGenerator : public PrimaryVertexGenerator {

public:
  /// Default constructor
  FlatPrimaryVertexGenerator(const edm::ParameterSet& vtx);

  /// Destructor
  ~FlatPrimaryVertexGenerator() {;}
  
  /// Generation process (to be implemented)
  virtual void generate(RandomEngineAndDistribution const*);

 private:

  // The smearing quantities in all three directions
  double minX;
  double minY;
  double minZ;
  double maxX;
  double maxY;
  double maxZ;

};

#endif // FlatPrimaryVertexGenerator_H

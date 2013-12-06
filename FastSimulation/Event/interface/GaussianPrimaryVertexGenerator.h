#ifndef FastSimulation_Event_GaussianPrimaryVertexGenerator_H
#define FastSimulation_Event_GaussianPrimaryVertexGenerator_H

// Famos Headers
#include "FastSimulation/Event/interface/PrimaryVertexGenerator.h"

/** A class that generates a primary vertex with a Gaussian profile, in cm*/ 

class RandomEngineAndDistribution;

namespace edm { 
  class ParameterSet;
}

class GaussianPrimaryVertexGenerator : public PrimaryVertexGenerator {

public:
  /// Default constructor
  GaussianPrimaryVertexGenerator(const edm::ParameterSet& vtx);

  /// Destructor
  ~GaussianPrimaryVertexGenerator() {;}
  
  /// Generation process (to be implemented)
  virtual void generate(RandomEngineAndDistribution const*);

 private:

  // The smearing quantities in all three directions
  double meanX;
  double meanY;
  double meanZ;
  double sigmaX;
  double sigmaY;
  double sigmaZ;

};

#endif // GaussianPrimaryVertexGenerator_H

#ifndef FastSimulation_Event_BetaFuncPrimaryVertexGenerator_H
#define FastSimulation_Event_BetaFuncPrimaryVertexGenerator_H

// Famos Headers
#include "FastSimulation/Event/interface/PrimaryVertexGenerator.h"

/** A class that generates a primary vertex with a realistic profile, in cm*/ 

class RandomEngine;

namespace edm { 
  class ParameterSet;
}

class BetaFuncPrimaryVertexGenerator : public PrimaryVertexGenerator {


public:
  /// Default constructor
  BetaFuncPrimaryVertexGenerator(const edm::ParameterSet& vtx,
				 const RandomEngine* engine);

  /// Destructor
  ~BetaFuncPrimaryVertexGenerator() {;}
  
  /// Generation process (to be implemented)
  virtual void generate();

  /// set resolution in Z in cm
  /// set mean in X in cm
  /// beta function
  double BetaFunction(double z, double z0);
  
private:

  TMatrixD* inverseLorentzBoost();
  
  double fX0, fY0, fZ0;
  double fSigmaZ;
  double alpha_, phi_;
  double fbetastar, femittance;
  
};

#endif // BetaFuncPrimaryVertexGenerator_H

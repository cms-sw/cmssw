#ifndef FastSimulation_MaterialEffects_NuclearInteractionEDMUpdator_H
#define FastSimulation_MaterialEffects_NuclearInteractionEDMUpdator_H

/** 
 * This class computes the probability for hadrons to interact with a 
 * nucleon of the tracker material (inelastically) and then reads a 
 * nuclear interaction randomly from multiple fully simulated files 
 * The fraction of interaction lengths traversed by the particle in this 
 * tracker layer is determined in MaterialEffectsUpdator as a fraction 
 * the radiation lengths. 
 *
 * \author Patrick Janot
 * $Date: 25-Jan-2007
 */ 

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/VectorInputSource.h"

#include "FastSimulation/MaterialEffects/interface/MaterialEffectsUpdator.h"

#include <vector>
#include <string>

class TFile;
class TTree;
class TBranch;
class NUEvent;

namespace edm { 
  class ModuleDescription;
}

class ParticlePropagator;
class RandomEngine;

class NuclearInteractionEDMUpdator : public MaterialEffectsUpdator
{
 public:

  /// Constructor
  NuclearInteractionEDMUpdator(const edm::ParameterSet& inputFiles,
			       std::vector<double>& pionEnergies,
			       double pionEnergy,
			       double lengthRatio,
			       const RandomEngine* engine);

  /// Default Destructor
  ~NuclearInteractionEDMUpdator();

 private:

  /// Generate an e+e- pair according to the probability that it happens
  void compute(ParticlePropagator& Particle);

  typedef edm::VectorInputSource::EventPrincipalVector EventPrincipalVector;
  edm::VectorInputSource* const input;
  edm::ModuleDescription md_;

  std::vector<double> thePionCM;
  double thePionEnergy;
  double theLengthRatio;

};
#endif

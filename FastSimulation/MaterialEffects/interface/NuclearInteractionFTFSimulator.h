#ifndef FastSimulation_MaterialEffects_NuclearInteractionFTFSimulator_H
#define FastSimulation_MaterialEffects_NuclearInteractionFTFSimulator_H

/** 
 * This class computes the probability for hadrons to interact with a 
 * nucleon of the tracker material (inelastically) and then sample
 * nuclear interaction using FTF model of Geant4 
 * The fraction of interaction lengths traversed by the particle in this 
 * tracker layer is determined in MaterialEffectsSimulator as a fraction 
 * the radiation lengths. 
 *
 * \author Vladimir Ivanchenko
 * $Date: 20-Jan-2015
 */ 

#include "FastSimulation/MaterialEffects/interface/MaterialEffectsSimulator.h"

#include "G4Nucleus.hh"
#include "G4HadProjectile.hh"
#include "G4LorentzVector.hh"
#include "G4ThreeVector.hh"

class ParticlePropagator;
class RandomEngineAndDistribution;
class G4ParticleDefinition;
class G4Track;
class G4Step;
class G4TheoFSGenerator;
class G4FTFModel;
class G4ExcitedStringDecay;
class G4LundStringFragmentation;
class G4GeneratorPrecompoundInterface;
class G4CascadeInterface;
class G4DiffuseElastic;
class G4PhysicsLogVector;

static const int numHadrons = 30;
static const int npoints = 21;

class NuclearInteractionFTFSimulator : public MaterialEffectsSimulator
{
public:

  /// Constructor
  NuclearInteractionFTFSimulator(unsigned int distAlgo, double distCut, 
				 double elimit, double eth);

  /// Default Destructor
  ~NuclearInteractionFTFSimulator();

private:

  /// Generate a nuclear interaction according to the probability that it happens
  void compute(ParticlePropagator& Particle, RandomEngineAndDistribution const*);

  void saveDaughter(ParticlePropagator& Particle, const G4LorentzVector& lv, int pdgid);

  double distanceToPrimary(const RawParticle& Particle,
			   const RawParticle& aDaughter) const;

  static const G4ParticleDefinition* theG4Hadron[numHadrons];
  static int theId[numHadrons];
  G4PhysicsLogVector* vect;

  G4TheoFSGenerator* theHadronicModel;
  G4FTFModel* theStringModel;
  G4ExcitedStringDecay* theStringDecay; 
  G4LundStringFragmentation* theLund;
  G4GeneratorPrecompoundInterface* theCascade; 

  G4CascadeInterface* theBertiniCascade;
  G4DiffuseElastic* theDiffuseElastic;

  G4Step* dummyStep;
  G4Track* currTrack;
  const G4ParticleDefinition* currParticle;

  G4Nucleus targetNucleus;
  G4HadProjectile theProjectile;
  G4LorentzVector curr4Mom;
  G4ThreeVector vectProj;
  G4ThreeVector theBoost;

  double theBertiniLimit;
  double theEnergyLimit;

  double theDistCut;
  double distMin;

  double intLengthElastic;
  double intLengthInelastic;

  int currIdx;
  size_t index;
  unsigned int theDistAlgo;
};
#endif

#ifndef FastSimulation_MaterialEffects_NuclearInteractionUpdator_H
#define FastSimulation_MaterialEffects_NuclearInteractionUpdator_H

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

#include "FastSimulation/MaterialEffects/interface/MaterialEffectsUpdator.h"

#include <vector>
#include <string>
#include <fstream>

class TFile;
class TTree;
class TBranch;
class NUEvent;

class ParticlePropagator;
class RandomEngine;
//class DaqMonitorBEInterface;
//class MonitorElement;

class NuclearInteractionUpdator : public MaterialEffectsUpdator
{
 public:

  /// Constructor
  NuclearInteractionUpdator(std::vector<std::string>& inputFiles,
			    std::vector<double>& pionEnergies,
			    double pionEnergy,
			    double lengthRatio,
			    std::vector<double> ratioRatio,
			    std::string inputFile,
			    const RandomEngine* engine);

  /// Default Destructor
  ~NuclearInteractionUpdator();

  /// Save current nuclear interaction (for later use)
  void save();

  /// Read former nuclear interaction (from previous run)
  void read(std::string inputFile);

 private:

  /// Generate a nuclear interaction according to the probability that it happens
  void compute(ParticlePropagator& Particle);

  std::vector<std::string> theFileNames;
  std::vector<double> thePionCM;
  double thePionEnergy;
  double theLengthRatio;
  std::vector<double> theRatios;

  std::vector<TFile*> theFiles;
  std::vector<TTree*> theTrees;
  std::vector<TBranch*> theBranches;
  std::vector<NUEvent*> theNUEvents;
  std::vector<unsigned> theCurrentEntry;
  std::vector<unsigned> theCurrentInteraction;
  std::vector<unsigned> theNumberOfEntries;
  std::vector<unsigned> theNumberOfInteractions;

  std::ofstream myOutputFile;
  unsigned myOutputBuffer;

  //  DaqMonitorBEInterface * dbe;
  //  MonitorElement* htot;
  //  MonitorElement* helas;
  //  MonitorElement* hinel;
  //  MonitorElement* hscatter;
  //  MonitorElement* hscatter2;

};
#endif

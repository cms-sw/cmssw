#ifndef FastSimulation_PileUpProducer_PileUpSimulator_H
#define FastSimulation_PileUpProducer_PileUpSimulator_H

/** 
 * This class delivers a number of minimum bias events, read from 
 * pre-produced event files, according to a Poisson distribution/ 
 *
 * \author Patrick Janot
 * $Date: 24-Apr-2007
 */ 

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>
#include <fstream>

class TFile;
class TTree;
class TBranch;
class PUEvent;

namespace edm { 
  class ParameterSet;
} 

namespace HepMC { 
  class GenEvent;
}

class FSimEvent;

class PileUpSimulator
{
 public:

  /// Constructor
  PileUpSimulator(FSimEvent* aSimEvent);

  /// Default Destructor
  ~PileUpSimulator();

  /// Produce N minimum bias events, and add them to the FSimEvent
  void produce(const HepMC::GenEvent* pu);

 private:

  double averageNumber_;
  FSimEvent* mySimEvent;
  std::vector<std::string> theFileNames;
  std::string inputFile;
  unsigned theNumberOfFiles;

  std::vector<TFile*> theFiles;
  std::vector<TTree*> theTrees;
  std::vector<TBranch*> theBranches;
  std::vector<PUEvent*> thePUEvents;
  std::vector<unsigned> theCurrentEntry;
  std::vector<unsigned> theCurrentMinBiasEvt;
  std::vector<unsigned> theNumberOfEntries;
  std::vector<unsigned> theNumberOfMinBiasEvts;

  std::ofstream myOutputFile;
  unsigned myOutputBuffer;

};
#endif

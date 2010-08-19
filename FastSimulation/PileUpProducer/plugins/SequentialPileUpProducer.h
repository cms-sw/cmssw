#ifndef FastSimulation_PileUpProducer_SequentialPileUpProducer_H
#define FastSimulation_PileUpProducer_SequentialPileUpProducer_H

#include "FWCore/Framework/interface/EDProducer.h"

#include <vector>
#include <string>
#include <fstream>

class ParameterSet;
class Event;
class EventSetup;

class TFile;
class TTree;
class TBranch;
class PUEvent;

class PrimaryVertexGenerator;
class RandomEngine;

class SequentialPileUpProducer : public edm::EDProducer
{

 public:

  explicit SequentialPileUpProducer(edm::ParameterSet const & p);
  virtual ~SequentialPileUpProducer();
  virtual void beginRun(edm::Run &, edm::EventSetup const&);
  virtual void endRun();
  virtual void produce(edm::Event & e, const edm::EventSetup & c);

 private:

  void openFile(unsigned file);

  const RandomEngine* random;
  PrimaryVertexGenerator* theVertexGenerator;

  double averageNumber_;
  unsigned theStartingEvent;

  std::vector<std::string> theFileNames;
  unsigned theNumberOfFiles;

  unsigned theNumberOfMinBiasEventsPerFile;

  bool skipSearchPath;

  TFile* theFile;
  TTree* theTree;
  TBranch* theBranch;
  PUEvent* thePUEvent;

  unsigned theNumberOfEntries;
  unsigned theNumberOfMinBiasEvts;
  unsigned theCurrentFile;
  unsigned theCurrentEntry;
  unsigned theCurrentMinBiasEvt;

};

#endif

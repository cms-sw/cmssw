#ifndef FastSimulation_PileUpProducer_PileUpProducer_H
#define FastSimulation_PileUpProducer_PileUpProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"

#include <vector>
#include <string>
#include <fstream>
#include "TH1F.h"

class ParameterSet;
class Event;
class EventSetup;

class TFile;
class TTree;
class TBranch;
class PUEvent;

class PrimaryVertexGenerator;
class RandomEngine;

class PileUpProducer : public edm::EDProducer
{

 public:

  explicit PileUpProducer(edm::ParameterSet const & p);
  virtual ~PileUpProducer();
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void produce(edm::Event & e, const edm::EventSetup & c) override;

 private:

  /// Save current minbias configuration (for later use)
  void save();

  /// Read former minbias configuration (from previous run)
  bool read(std::string inputFile);

 private:

  PrimaryVertexGenerator* theVertexGenerator;

  double averageNumber_;
  const RandomEngine* random;
  std::vector<std::string> theFileNames;
  std::string inputFile;
  unsigned theNumberOfFiles;
  bool usePoisson_;

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

  TH1F * hprob;
  std::vector<int> dataProbFunctionVar;
  std::vector<double> dataProb;
  int varSize;
  int probSize;
};

#endif

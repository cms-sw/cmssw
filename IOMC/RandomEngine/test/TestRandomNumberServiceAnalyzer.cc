// -*- C++ -*-
//
// Package: IOMC/RandomEngine
// Class: TestRandomNumberServiceAnalyzer
//
/**\class TestRandomNumberServiceAnalyzer

Description: Used in tests of the RandomNumberGeneratorService.

Implementation: Generates some random numbers using the engines
from the service. Prints them to output text files.

NOTE: This is only used to test the mode where we fork processes
(multiprocess mode not multithreaded mode).  It needs the
postForReacquireResources method. This method so far has
not been implemented for the "one", "global", or "stream"
type modules. Therefore this class has to remain as a
"classic" (aka "legacy" type module). The idea is that
the whole forking mode may be dropped once the multithreaded
mode is proved to work well. Then we can delete this module
and the test that uses it. If that doesn't happen and we want
to really eliminate/convert all "classic" type modules, then
someone needs to implement the postForReacquireResources method
for one of the other module types or change the design of this
test.

*/
//
// Original Author: Chris Jones, David Dagenhart
// Created: Tue Mar 7 11:57:09 EST 2006
//

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandExponential.h"
#include "CLHEP/Random/RandomEngine.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

class TestRandomNumberServiceAnalyzer : public edm::EDAnalyzer /* See comment above, there is a reason this is still a classic module */ {
public:
  explicit TestRandomNumberServiceAnalyzer(edm::ParameterSet const& pset);
  ~TestRandomNumberServiceAnalyzer();

  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) override;
  virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) override;
  virtual void endJob() override;

private:
  bool multiprocess_;
  unsigned childIndex_;
  unsigned count_;
  double randomNumberEvent0_;
  double randomNumberEvent1_;
  double randomNumberEvent2_;
  double randomNumberEvent3_;
  double randomNumberLumi0_;
  double randomNumberLumi1_;
  double randomNumberLumi2_;
  std::string lastEventRandomNumbers_;
};

TestRandomNumberServiceAnalyzer::TestRandomNumberServiceAnalyzer(edm::ParameterSet const& pset) :
  multiprocess_(false),
  childIndex_(0U),
  count_(0U),
  randomNumberEvent0_(0.0),
  randomNumberEvent1_(0.0),
  randomNumberEvent2_(0.0),
  randomNumberEvent3_(0.0),
  randomNumberLumi0_(0.0),
  randomNumberLumi1_(0.0),
  randomNumberLumi2_(0.0) {
}

TestRandomNumberServiceAnalyzer::~TestRandomNumberServiceAnalyzer() {
}

void
TestRandomNumberServiceAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const&) {

  // Add some sleep for the different child processes in attempt
  // to ensure all the child processes get events to process.
  if(multiprocess_) {
    sleep(0.025 + childIndex_ * 0.025 + count_ * 0.3);
  }
  ++count_;

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine(iEvent.streamID());

  randomNumberEvent0_ = engine.flat();
  randomNumberEvent1_ = engine.flat();
  randomNumberEvent2_ = engine.flat();

  CLHEP::RandExponential expDist(engine);
  double mean = 10.0; // Mean of the exponential
  randomNumberEvent3_ = expDist.fire(mean);

  // Print some random numbers from the first event
  if(multiprocess_ && count_ == 1U) {

    std::ostringstream ss;
    ss << "child" << childIndex_ << "FirstEvent.txt";
    std::string filename = ss.str();

    std::ofstream outFile;

    outFile.open(filename.c_str(), std::ofstream::app);
    
    outFile << moduleDescription().moduleLabel();
    outFile << " Event random numbers ";
    outFile << randomNumberEvent0_ << " ";
    outFile << randomNumberEvent1_ << " ";
    outFile << randomNumberEvent2_ << " ";
    outFile << randomNumberEvent3_ << " ";

    outFile << "Lumi random numbers ";
    outFile << randomNumberLumi0_ << " ";
    outFile << randomNumberLumi1_ << " ";
    outFile << randomNumberLumi2_ << "\n";

    outFile.close();
  }

  // Save a string with some random numbers, overwrite at each
  // event so at endJob this will only contain content from the last event

  std::ostringstream ss;

  ss << moduleDescription().moduleLabel();
  ss << " Event random numbers ";
  ss << randomNumberEvent0_ << " ";
  ss << randomNumberEvent1_ << " ";
  ss << randomNumberEvent2_ << " ";
  ss << randomNumberEvent3_ << " ";

  ss << "Lumi random numbers ";
  ss << randomNumberLumi0_ << " ";
  ss << randomNumberLumi1_ << " ";
  ss << randomNumberLumi2_ << "\n";

  lastEventRandomNumbers_ = ss.str();

  // Print some random numbers each event
  std::ostringstream ss1;
  ss1 << "testRandomServiceL" << iEvent.eventAuxiliary().luminosityBlock()
      << "E" << iEvent.eventAuxiliary().event() << ".txt";
  std::string filename = ss1.str();

  std::ofstream outFile;

  outFile.open(filename.c_str(), std::ofstream::app);

  outFile << moduleDescription().moduleLabel();
  outFile << " Event random numbers ";
  outFile << randomNumberEvent0_ << " ";
  outFile << randomNumberEvent1_ << " ";
  outFile << randomNumberEvent2_ << " ";
  outFile << randomNumberEvent3_ << " ";

  outFile << "Lumi random numbers ";
  outFile << randomNumberLumi0_ << " ";
  outFile << randomNumberLumi1_ << " ";
  outFile << randomNumberLumi2_ << "\n";

  outFile.close();
}

void TestRandomNumberServiceAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine(lumi.index());

  randomNumberLumi0_ = engine.flat();
  randomNumberLumi1_ = engine.flat();
  randomNumberLumi2_ = engine.flat();
}

void TestRandomNumberServiceAnalyzer::postForkReacquireResources(unsigned int iChildIndex, unsigned int /*iNumberOfChildren*/) {
  multiprocess_ = true;
  childIndex_ = iChildIndex;
}

void TestRandomNumberServiceAnalyzer::endJob() {

  std::ostringstream ss;
  ss << "child" << childIndex_ << "LastEvent.txt";
  std::string filename = ss.str();

  std::ofstream outFile;
  outFile.open(filename.c_str(), std::ofstream::app);

  outFile << lastEventRandomNumbers_;

  outFile.close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestRandomNumberServiceAnalyzer);

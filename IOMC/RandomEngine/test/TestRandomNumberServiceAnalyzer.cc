// -*- C++ -*-
//
// Package:    IOMC/RandomEngine
// Class:      TestRandomNumberServiceAnalyzer
//
/**\class TestRandomNumberServiceAnalyzer TestRandomNumberServiceAnalyzer.cc IOMC/RandomEngine/test/TestRandomNumberServiceAnalyzer.cc

 Description: Used in tests of the RandomNumberGeneratorService.

 Implementation: Generates some random numbers using the engines from the
service.  Prints them to an output file named testRandomNumberService.txt.
*/
//
// Original Author:  Chris Jones, David Dagenhart
//         Created:  Tue Mar  7 11:57:09 EST 2006
//

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
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

class TestRandomNumberServiceAnalyzer : public edm::EDAnalyzer {
  public:
    explicit TestRandomNumberServiceAnalyzer(edm::ParameterSet const& pset);
    ~TestRandomNumberServiceAnalyzer();

    virtual void analyze(edm::Event const& ev, edm::EventSetup const& es) override;
    virtual void beginJob();
    virtual void endJob();
    virtual void beginRun(edm::Run const& run, edm::EventSetup const& es) override;
    virtual void endRun(edm::Run const& run, edm::EventSetup const& es) override;
    virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) override;
    virtual void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) override;
    virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

  private:
    static bool firstFileOpen_;
    bool dump_;
    std::string outFileName_;
    bool multiprocess_;
    unsigned childIndex_;
    unsigned count_;
    bool firstInPath_;
    double randomNumberEvent0_;
    double randomNumberEvent1_;
    double randomNumberEvent2_;
    double randomNumberEvent3_;
    double randomNumberLumi0_;
    double randomNumberLumi1_;
    double randomNumberLumi2_;
    bool multiprocessReplay_;
};

bool TestRandomNumberServiceAnalyzer::firstFileOpen_ = true;

TestRandomNumberServiceAnalyzer::TestRandomNumberServiceAnalyzer(edm::ParameterSet const& pset) :
  dump_(pset.getUntrackedParameter<bool>("dump", false)),
  outFileName_("testRandomService.txt"),
  multiprocess_(false),
  childIndex_(0U),
  count_(0U),
  firstInPath_(pset.getUntrackedParameter<bool>("firstInPath", false)),
  randomNumberEvent0_(0.0),
  randomNumberEvent1_(0.0),
  randomNumberEvent2_(0.0),
  randomNumberEvent3_(0.0),
  randomNumberLumi0_(0.0),
  randomNumberLumi1_(0.0),
  randomNumberLumi2_(0.0),
  multiprocessReplay_(pset.getUntrackedParameter<bool>("multiprocessReplay", false)) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if(dump_) {
    rng->print();
    std::cout << "*** TestRandomNumberServiceAnalyzer constructor " << rng->mySeed() << "\n";
  }
  if(rng->getEngine().name() !="RanecuEngine") {
     //std::cout <<rng->getEngine().name()<<" "<<rng->mySeed()<<" "<<rng->getEngine().getSeed()<<std::endl;
     assert(rng->mySeed()==rng->getEngine().getSeed());
  }
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
  if(dump_) {
    std::cout << "*** TestRandomNumberServiceAnalyzer analyze " << rng->mySeed() << "\n";
  }

  std::ofstream outFile;
  outFile.open(outFileName_.c_str(), std::ofstream::out | std::ofstream::app);

  outFile << "*** TestRandomNumberServiceAnalyzer analyze() "
          << iEvent.eventAuxiliary().run()
          << "/" << iEvent.eventAuxiliary().luminosityBlock()
          << "/" << iEvent.eventAuxiliary().event()
          << "\n";
  outFile << rng->mySeed() << "\n";
  outFile << rng->getEngine().name() << "\n";
  
  // Get a reference to the engine.  This call can
  // be here or it can be in the module constructor
  // if the class saves the reference as a member.  It is
  // important that users NOT directly reset the seeds in
  // the engine, reset the state of the engine, or try
  // to destroy the engine object.  The service takes
  // care of those actions.
  CLHEP::HepRandomEngine& engine = rng->getEngine();

  // Generate random numbers distributed flatly between 0 and 1
  randomNumberEvent0_ = engine.flat();
  randomNumberEvent1_ = engine.flat();
  randomNumberEvent2_ = engine.flat();

  outFile << randomNumberEvent0_ << "\n";
  outFile << randomNumberEvent1_ << "\n";
  outFile << randomNumberEvent2_ << "\n";

  // An example of how to generate random numbers using the distributions
  // in CLHEP.  Here we use the exponential distribution.  CLHEP provides
  // 9 other possible distributions.
  // It is very important to use the "fire" method not "shoot".
  CLHEP::RandExponential expDist(engine);
  double mean = 10.0;  // Mean of the exponential

  randomNumberEvent3_ = expDist.fire(mean);
  outFile << randomNumberEvent3_ << "\n";

  outFile.close();

  if(multiprocess_ && count_ == 1U) {
    std::ostringstream ss;
    ss << "child" << childIndex_ << "FirstEvent.txt";
    std::string filename = ss.str();

    if(firstInPath_) {
      outFile.open(filename.c_str());
    } else {
      outFile.open(filename.c_str(), std::ofstream::app);
    }
    outFile << *currentContext()->moduleLabel() << "\n";
    outFile << rng->mySeed() << "\n";
    outFile << rng->getEngine().name() << "\n";

    outFile << "Event random numbers\n";
    outFile << randomNumberEvent0_ << "\n";
    outFile << randomNumberEvent1_ << "\n";
    outFile << randomNumberEvent2_ << "\n";
    outFile << randomNumberEvent3_ << "\n";

    outFile << "Lumi random numbers\n";
    outFile << randomNumberLumi0_ << "\n";
    outFile << randomNumberLumi1_ << "\n";
    outFile << randomNumberLumi2_ << "\n";

    outFile.close();
  }

  if(multiprocess_ || multiprocessReplay_) {
    std::ostringstream ss;
    ss << "child" << childIndex_ << "LastEvent.txt";
    std::string filename = ss.str();

    if(firstInPath_) {
      outFile.open(filename.c_str());
    } else {
      outFile.open(filename.c_str(), std::ofstream::app);
    }
    outFile << *currentContext()->moduleLabel() << "\n";
    outFile << rng->mySeed() << "\n";
    outFile << rng->getEngine().name() << "\n";

    outFile << "Event random numbers\n";
    outFile << randomNumberEvent0_ << "\n";
    outFile << randomNumberEvent1_ << "\n";
    outFile << randomNumberEvent2_ << "\n";
    outFile << randomNumberEvent3_ << "\n";

    outFile << "Lumi random numbers\n";
    outFile << randomNumberLumi0_ << "\n";
    outFile << randomNumberLumi1_ << "\n";
    outFile << randomNumberLumi2_ << "\n";

    outFile.close();
  }

  if(multiprocess_ || multiprocessReplay_) {
    std::ostringstream ss;
    ss << "testRandomServiceL" << iEvent.eventAuxiliary().luminosityBlock()
       << "E" << iEvent.eventAuxiliary().event() << ".txt";
    std::string filename = ss.str();

    if(firstInPath_) {
      outFile.open(filename.c_str());
    } else {
      outFile.open(filename.c_str(), std::ofstream::app);
    }
    outFile << *currentContext()->moduleLabel() << "\n";
    outFile << rng->mySeed() << "\n";
    outFile << rng->getEngine().name() << "\n";

    outFile << "Event random numbers\n";
    outFile << randomNumberEvent0_ << "\n";
    outFile << randomNumberEvent1_ << "\n";
    outFile << randomNumberEvent2_ << "\n";
    outFile << randomNumberEvent3_ << "\n";

    outFile << "Lumi random numbers\n";
    outFile << randomNumberLumi0_ << "\n";
    outFile << randomNumberLumi1_ << "\n";
    outFile << randomNumberLumi2_ << "\n";

    outFile.close();
  }
}

void TestRandomNumberServiceAnalyzer::beginJob() {
  edm::Service<edm::RandomNumberGenerator> rng;
  if(dump_) {
    std::cout << "*** TestRandomNumberServiceAnalyzer beginJob " << rng->mySeed() << "\n";
    std::cout << rng->getEngine().name() << "\n";
  }
}

void TestRandomNumberServiceAnalyzer::endJob() {
  edm::Service<edm::RandomNumberGenerator> rng;
  if(dump_) {
    std::cout << "*** TestRandomNumberServiceAnalyzer endJob " << rng->mySeed() << "\n";
  }
}

void TestRandomNumberServiceAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if(dump_) {
    std::cout << "*** TestRandomNumberServiceAnalyzer beginRun " << rng->mySeed() << "\n";
  }
}

void TestRandomNumberServiceAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if(dump_) {
    std::cout << "*** TestRandomNumberServiceAnalyzer endRun " << rng->mySeed() << "\n";
  }
}

void TestRandomNumberServiceAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if(dump_) {
    std::cout << "*** TestRandomNumberServiceAnalyzer beginLuminosityBlock " << rng->mySeed() << "\n";
  }

  // The first time we open the file we create a new file
  // After that append to it.  This file is just for testing
  // purposes, we print out the generated random numbers and
  // some other things.
  std::ofstream outFile;
  if(firstFileOpen_) {
    outFile.open(outFileName_.c_str());
    firstFileOpen_ = false;
  } else {
    outFile.open(outFileName_.c_str(), std::ofstream::out | std::ofstream::app);
  }

  outFile << "*** TestRandomNumberServiceAnalyzer beginLumi " << lumi.luminosityBlockAuxiliary().run()
          << "/" << lumi.luminosityBlockAuxiliary().luminosityBlock()  << "\n";
  outFile << rng->mySeed() << "\n";
  outFile << rng->getEngine().name() << "\n";

  CLHEP::HepRandomEngine& engine = rng->getEngine();

  // Generate random numbers distributed flatly between 0 and 1
  randomNumberLumi0_ = engine.flat();
  randomNumberLumi1_ = engine.flat();
  randomNumberLumi2_ = engine.flat();

  outFile << randomNumberLumi0_ << "\n";
  outFile << randomNumberLumi1_ << "\n";
  outFile << randomNumberLumi2_ << "\n";

  outFile.close();
}

void TestRandomNumberServiceAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if(dump_) {
    std::cout << "*** TestRandomNumberServiceAnalyzer endLuminosityBlock " << rng->mySeed() << "\n";
  }
}

void TestRandomNumberServiceAnalyzer::postForkReacquireResources(unsigned int iChildIndex, unsigned int /*iNumberOfChildren*/) {
  multiprocess_ = true;
  childIndex_ = iChildIndex;
  std::ostringstream suffix;
  suffix << "_" << iChildIndex;
  outFileName_ = std::string("testRandomService") + suffix.str() + std::string(".txt");
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestRandomNumberServiceAnalyzer);

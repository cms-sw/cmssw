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
// $Id$
//
//

#include <memory>
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandExponential.h"


class TestRandomNumberServiceAnalyzer : public edm::EDAnalyzer {
  public:
    explicit TestRandomNumberServiceAnalyzer(const edm::ParameterSet&);
    ~TestRandomNumberServiceAnalyzer();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

  private:
    static bool firstFileOpen_;
};

bool TestRandomNumberServiceAnalyzer::firstFileOpen_ = true;


TestRandomNumberServiceAnalyzer::TestRandomNumberServiceAnalyzer(const edm::ParameterSet& iConfig)
{

  // The first time we open the file we create a new file
  // After that append to it.  This file is just for testing
  // purposes, we print out the generated random numbers and
  // some other things.
  std::ofstream outFile;
  if (firstFileOpen_) {
    outFile.open("testRandomService.txt");
    firstFileOpen_ = false;
  }
  else {
    outFile.open("testRandomService.txt", std::ofstream::out | std::ofstream::app); 
  }

  // The rest of the code in this function gets repeated in
  //the analyze function, there are comments there.

  using namespace edm;
  Service<RandomNumberGenerator> rng;

  outFile << "*** TestRandomNumberServiceAnalyzer constructor\n";

  // rng->print();

  outFile << rng->mySeed() << "\n";

  CLHEP::HepRandomEngine& engine = rng->getEngine();
  for (int i = 0; i < 5; ++i) { 
    double num = engine.flat();
    outFile << num << "\n";
  }
  outFile.close();
}


TestRandomNumberServiceAnalyzer::~TestRandomNumberServiceAnalyzer()
{
}


void
TestRandomNumberServiceAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::ofstream outFile;
  outFile.open("testRandomService.txt", std::ofstream::out | std::ofstream::app); 

  // Get the service
  using namespace edm;
  Service<RandomNumberGenerator> rng;

  outFile << "*** TestRandomNumberServiceAnalyzer analyze() ***\n";

  // This is useful for debugging but normally leave it
  // commented out because it sends lots of output to std::cout
  // It prints out the internal state of the service
  // rng->print();

  // The first random seed
  outFile << rng->mySeed() << "\n";

  // Get a reference to the engine.  This call can
  // be here or it can be in the module constructor
  // if the class saves the reference as a member.  It is
  // important that users NOT directly reset the seeds in
  // the engine, reset the state of the engine, or try
  // to destroy the engine object.  The service takes
  // care of those actions.
  CLHEP::HepRandomEngine& engine = rng->getEngine();

  double randomNumber;

  // Generate random numbers distributed flatly between 0 and 1 
  for (int i = 0; i < 5; ++i) { 
    randomNumber = engine.flat();
    outFile << randomNumber << "\n";
  }

  // An example of how to generate random numbers using the distributions
  // in CLHEP.  Here we use the exponential distribution.  CLHEP provides
  // 9 other possible distributions.
  // It is very important to use the "fire" method not "shoot".
  RandExponential expDist(engine);
  double mean = 10.0;  // Mean of the exponential
  for (int i = 0; i < 5; ++i) {
    randomNumber = expDist.fire(mean);
    outFile << randomNumber << "\n";
  }

  outFile.close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestRandomNumberServiceAnalyzer);

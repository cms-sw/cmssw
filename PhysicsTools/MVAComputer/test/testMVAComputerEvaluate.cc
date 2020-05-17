#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"

#include "PhysicsTools/MVAComputer/test/testMVAComputerEvaluate.h"

// take the event setup record for "MVADemoRcd" from the header above
// definition shared with PhysicsTools/MVATrainer/test/testMVATrainerLooper
// (the "Rcd" is implicitly appended by the macro)
//
// MVA_COMPUTER_CONTAINER_DEFINE(MVADemo);

using namespace PhysicsTools;

class testMVAComputerEvaluate : public edm::EDAnalyzer {
public:
  explicit testMVAComputerEvaluate(const edm::ParameterSet& params);

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  MVAComputerCache mvaComputer;
};

testMVAComputerEvaluate::testMVAComputerEvaluate(const edm::ParameterSet& params) {}

void testMVAComputerEvaluate::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // update the cached MVAComputer from calibrations
  // passed via EventSetup.
  // you can use a MVAComputerContainer to pass around
  // multiple different MVA's in one event setup record
  // identify the right one by a definable name string
  mvaComputer.update<MVADemoRcd>(iSetup, "testMVA");

  Variable::Value values[] = {Variable::Value("x", 1.0), Variable::Value("y", 1.5)};

  double result = mvaComputer->eval(values, values + 2);
  // arguments are begin() and end() (for plain C++ arrays done this way)
  // std::vector also works, but plain array has better performance
  // for fixed-size arrays (no internal malloc/free)

  std::cout << "mva.eval(x = 1.0, y = 1.5) = " << result << std::endl;
}

// define this as a plug-in
DEFINE_FWK_MODULE(testMVAComputerEvaluate);

// define the plugins for the record
MVA_COMPUTER_CONTAINER_IMPLEMENT(MVADemo);
// this will implictly define an EDM es_source named "MVADemoFileSource"
// which will allow to read the calibration from file into the EventSetup
// note that for CondDB the PoolDBESSource can be used instead

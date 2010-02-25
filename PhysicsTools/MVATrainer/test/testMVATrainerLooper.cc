#include <iostream>
#include <memory>

#include <TRandom.h>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#include "PhysicsTools/MVATrainer/interface/HelperMacros.h"

#include "PhysicsTools/MVAComputer/test/testMVAComputerEvaluate.h"

// take the event setup record for "MVADemoRcd" from the header above
// definition shared with PhysicsTools/MVAComputer/test/testMVAComputerEvaluate
// (the "Rcd" is implicitly appended by the macro)
//
// MVA_COMPUTER_CONTAINER_DEFINE(MVADemo);

using namespace PhysicsTools;

class testMVATrainerLooper : public edm::EDAnalyzer {
    public:
	explicit testMVATrainerLooper(const edm::ParameterSet &params);

	virtual void beginRun(const edm::Run &run, const edm::EventSetup &iSetup);

	virtual void analyze(const edm::Event& iEvent,
	                     const edm::EventSetup& iSetup);

    private:
	MVAComputerCache		mvaComputer;
};

testMVATrainerLooper::testMVATrainerLooper(const edm::ParameterSet &params)
{
}

void testMVATrainerLooper::beginRun(const edm::Run &run, const edm::EventSetup &iSetup)
{
	// reset the random number generator here
	// we are expected to feed the same training events for
	// each loop iteration. We normally run from a ROOT file,
	// so we wouldn't need to care... (don't do this with real data!)

	gRandom->SetSeed(12345);
}

void testMVATrainerLooper::analyze(const edm::Event& iEvent,
                                   const edm::EventSetup& iSetup)
{
	// Note that the code here is almost IDENTICAL
	// to the code when just evaluating the MVA
	// The only differences are:
	// * EventSetup::get is called with additional argument "trainer"
	// * the variables contain a value for kTargetId with the truth
	// * the result of MVAComputer::eval is to be ignored
	//
	// So: When possible try to share the filling routine!

	// update the cached MVAComputer from calibrations
	// passed via EventSetup.
	// you can use a MVAComputerContainer to pass around
	// multiple different MVA's in one event setup record
	// identify the right one by a definable name string
	mvaComputer.update<MVADemoRcd>("trainer", iSetup, "testMVA");

	// can occur in last iteration, when MVATrainer has completed
	// and is about to save the result
	if (!mvaComputer)
		return;

	// Produce some random values to train on

	bool target = gRandom->Uniform() > 0.5; // true = signal, false = bkg
	double x, y;

	if (target) {
		x = gRandom->Gaus(+2, 2);
		y = gRandom->Gaus(+1, 2);
	} else {
		x = gRandom->Gaus(-1, 2);
		y = gRandom->Gaus(-2, 2);
	}

	Variable::Value values[] = {
		Variable::Value(MVATrainer::kTargetId, target),
	//	Variable::Value(MVATrainer::kWeightId, 1.0) // default = 1
		Variable::Value("x", x),
		Variable::Value("y", y)
	};

	mvaComputer->eval(values, values + 3);
	// arguments are begin() and end() (for plain C++ arrays done this way)
	// std::vector also works, but plain array has better performance
	// for fixed-size arrays (no internal malloc/free)

	// mvaComputer->eval() can be called as many times per event as needed
}

// define this as a plug-in
DEFINE_FWK_MODULE(testMVATrainerLooper);

// Here come the definition(s) for the CMSSW interaction:

// define the plugins for the trainer
MVA_TRAINER_IMPLEMENT(MVADemo);
// this will implictly define a bunch of EDM plugins:
// * module "MVADemoContainerSaveCondDB"
// * module "MVADemoSaveFile"
// * looper "MVADemoTrainerLooper"
//
// - the looper is mandatory, it will read the .xml steering file and
//   provide the training mechanism via EventSetup
//
// - for saving the result, either a standalone .mva file can be written
//   via MVADemoSaveFile, see testMVATrainerLooper.cfg
//
// - or to the conditions database via the other module
//   (in conjuction with PoolDBOutputService)


/**********************************************************************
 *
 * ATTENTION: instead off calling this here, please link against
 *            the library defining the record in the first place instead!
 *
 *            (We can't do it here, since we would need to link against
 *             a file from the MVAComputer/test directory, which is
 *             not possible in example code)
 *
 */	EVENTSETUP_RECORD_REG(MVADemoRcd);

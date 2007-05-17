#include <assert.h>
#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooper.h"

namespace PhysicsTools {

MVATrainerLooper::MVATrainerLooper(const edm::ParameterSet& iConfig)
{
	std::string trainFile =
			iConfig.getParameter<std::string>("trainDescription");

	trainer = std::auto_ptr<MVATrainer>(new MVATrainer(trainFile));
}

void MVATrainerLooper::startingNewLoop(unsigned int iteration)
{
	updateTrainer();
}

edm::EDLooper::Status
MVATrainerLooper::duringLoop(const edm::Event &event,
                                   const edm::EventSetup &es)
{
	return trainCalib ? kContinue : kStop;
}

edm::EDLooper::Status MVATrainerLooper::endOfLoop(const edm::EventSetup &es,
                                                  unsigned int iteration)
{
	updateTrainer();
	return trainCalib ? kContinue : kStop;
}

void MVATrainerLooper::updateTrainer()
{
	assert(trainCalib.use_count() <= 1);
	trainCalib.reset();
	trainCalib = TrainObject(trainer->getTrainCalibration());
}

} // namespace PhysicsTools

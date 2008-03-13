#include <assert.h>
#include <algorithm>
#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooper.h"

namespace PhysicsTools {

namespace {
	template<typename T>
	struct deleter : public std::unary_function<T*, void> {
		inline void operator() (T *ptr) const { delete ptr; }
	};
}

// MVATrainerLooper::Trainer implementation

MVATrainerLooper::Trainer::Trainer(const edm::ParameterSet &params)
{
	std::string trainDescription =
			params.getUntrackedParameter<std::string>(
							"trainDescription");
	bool doLoad = params.getUntrackedParameter<bool>("loadState", false);
	bool doSave = params.getUntrackedParameter<bool>("saveState", false);
	bool doMonitoring = params.getUntrackedParameter<bool>("monitoring", false);

	trainer = std::auto_ptr<MVATrainer>(new MVATrainer(trainDescription));

	if (doLoad)
		trainer->loadState();

	trainer->setAutoSave(doSave);
	trainer->setCleanup(!doSave);
	trainer->setMonitoring(doMonitoring);
}

// MVATrainerLooper::MVATrainerContainer implementation

MVATrainerLooper::TrainerContainer::~TrainerContainer()
{
	clear();
}

void MVATrainerLooper::TrainerContainer::clear()
{
	std::for_each(begin(), end(), deleter<Trainer>());
	content.clear();
}

// MVATrainerLooper implementation

MVATrainerLooper::MVATrainerLooper(const edm::ParameterSet& iConfig)
{
}

MVATrainerLooper::~MVATrainerLooper()
{
}

void MVATrainerLooper::startingNewLoop(unsigned int iteration)
{
	for(TrainerContainer::const_iterator iter = trainers.begin();
	    iter != trainers.end(); iter++) {
		Trainer *trainer = *iter;

		trainer->trainCalib =
			TrainObject(trainer->trainer->getTrainCalibration());
	}
}

edm::EDLooper::Status
MVATrainerLooper::duringLoop(const edm::Event &event,
                                   const edm::EventSetup &es)
{
	if (trainers.empty())
		return kStop;

	for(TrainerContainer::const_iterator iter = trainers.begin();
	    iter != trainers.end(); iter++)
		if ((*iter)->getCalibration())
			return kContinue;

	trainers.clear();
	return kStop;
}

edm::EDLooper::Status MVATrainerLooper::endOfLoop(const edm::EventSetup &es,
                                                  unsigned int iteration)
{
	if (trainers.empty())
		return kStop;

	for(TrainerContainer::const_iterator iter = trainers.begin();
	    iter != trainers.end(); iter++) {
		Trainer *trainer = *iter;

		if (trainer->trainCalib)
			trainer->trainer->doneTraining(
						trainer->trainCalib.get());

		trainer->trainCalib.reset();
	}

	return kContinue;
}

} // namespace PhysicsTools

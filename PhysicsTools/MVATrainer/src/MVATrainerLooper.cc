#include <cassert>
#include <algorithm>
#include <string>
#include <memory>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooper.h"

namespace PhysicsTools {

namespace {
    template<typename T>
    inline void deleter (T *ptr) { delete ptr; }
}

// MVATrainerLooper::Trainer implementation

MVATrainerLooper::Trainer::Trainer(const edm::ParameterSet &params)
{
	const edm::Entry *entry = params.retrieveUntracked("trainDescription");
	if (!entry)
		throw edm::Exception(edm::errors::Configuration,
		                     "MissingParameter:")
			<< "The required parameter 'trainDescription' "
			   "was not specified." << std::endl;;
	std::string trainDescription;
	if (entry->typeCode() == 'F')
		trainDescription = entry->getFileInPath().fullPath();
	else
		trainDescription = entry->getString();

	bool useXSLT = params.getUntrackedParameter<bool>("useXSLT", false);
	bool doLoad = params.getUntrackedParameter<bool>("loadState", false);
	bool doSave = params.getUntrackedParameter<bool>("saveState", false);
	bool doMonitoring = params.getUntrackedParameter<bool>("monitoring", false);

	trainer.reset(new MVATrainer(trainDescription, useXSLT));

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
	std::for_each(begin(), end(), deleter<Trainer>);
	content.clear();
}

// MVATrainerLooper implementation

MVATrainerLooper::MVATrainerLooper(const edm::ParameterSet& iConfig) :
  dataProcessedInLoop(false)
{
}

MVATrainerLooper::~MVATrainerLooper()
{
}

void MVATrainerLooper::startingNewLoop(unsigned int iteration)
{
        dataProcessedInLoop = false; 

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
        dataProcessedInLoop = true;

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
        if (!dataProcessedInLoop) {
          cms::Exception ex("MVATrainerLooper");
          ex << "No data processed during loop\n";
          ex.addContext("Calling MVATrainerLooper::endOfLoop()");
          throw ex;
        }

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

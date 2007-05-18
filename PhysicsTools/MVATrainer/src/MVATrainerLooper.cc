#include <assert.h>
#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/Utilities/interface/Exception.h"
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
	if (trainCalib)
		return kContinue;

	std::auto_ptr<Calibration::MVAComputer> calib =
				std::auto_ptr<Calibration::MVAComputer>(
						trainer->getCalibration());
	if (calib.get())
		storeCalibration(calib);
	else
		throw cms::Exception("MVATrainerLooper")
			<< "No calibration object obtained." << std::endl;

	return kStop;
}

void MVATrainerLooper::updateTrainer()
{
	if (trainCalib)
		trainer->doneTraining(trainCalib.get());
	trainCalib = TrainObject(trainer->getTrainCalibration());
}

} // namespace PhysicsTools

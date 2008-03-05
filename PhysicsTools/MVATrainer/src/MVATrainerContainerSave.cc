#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainerLooper.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerContainerSave.h"

namespace PhysicsTools {

MVATrainerContainerSave::MVATrainerContainerSave(
					const edm::ParameterSet &params) :
	toPut(params.getParameter<std::vector<std::string> >("toPut")),
	toCopy(params.getParameter<std::vector<std::string> >("toCopy")),
	saved(false)
{
}

void MVATrainerContainerSave::analyze(const edm::Event& event,
                                      const edm::EventSetup& es)
{
	if (calib.get() || saved)
		return;

	const Calibration::MVAComputerContainer *toPutCalib = 0;
	if (!toPut.empty()) {
		toPutCalib = getToPut(es);
		if (MVATrainerLooper::isUntrained(toPutCalib))
			return;
	}

	const Calibration::MVAComputerContainer *toCopyCalib = 0;
	if (!toCopy.empty())
		toCopyCalib = getToCopy(es);

	edm::LogInfo("MVATrainerSave") << "Got the trained calibration data";

	std::auto_ptr<Calibration::MVAComputerContainer> calib(
					new Calibration::MVAComputerContainer);

	for(std::vector<std::string>::const_iterator iter = toCopy.begin();
	    iter != toCopy.end(); iter++)
		calib->add(*iter) = toCopyCalib->find(*iter);

	for(std::vector<std::string>::const_iterator iter = toPut.begin();
	    iter != toPut.end(); iter++)
		calib->add(*iter) = toPutCalib->find(*iter);

	this->calib = calib;
}

void MVATrainerContainerSave::endJob()
{
	if (!calib.get() || saved)
		return;

	edm::LogInfo("MVATrainerSave") << "Saving calibration data in CondDB.";

	edm::Service<cond::service::PoolDBOutputService> dbService;
	if (!dbService.isAvailable())
		throw cms::Exception("MVATrainerContainerSave")
			<< "DBService unavailable" << std::endl;

	dbService->createNewIOV<Calibration::MVAComputerContainer>(
		calib.release(), dbService->beginOfTime(),
		dbService->endOfTime(), getRecordName().c_str());

	saved = true;
}

} // namespace PhysicsTools

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
#include "PhysicsTools/MVATrainer/interface/MVATrainerSave.h"

namespace PhysicsTools {

MVATrainerSave::MVATrainerSave(const edm::ParameterSet &params) :
	saved(false)
{
}

void MVATrainerSave::analyze(const edm::Event& event,
                             const edm::EventSetup& es)
{
	if (calib.get() || saved)
		return;

	const Calibration::MVAComputer *toPutCalib = getToPut(es);
	if (MVATrainerLooper::isUntrained(toPutCalib))
		return;

	edm::LogInfo("MVATrainerSave") << "Got the trained calibration data";

	std::auto_ptr<Calibration::MVAComputer> calib(
						new Calibration::MVAComputer);
	*calib = *toPutCalib;

	this->calib = calib;
}

void MVATrainerSave::endJob()
{
	if (!calib.get() || saved)
		return;

	edm::LogInfo("MVATrainerSave") << "Saving calibration data in CondDB.";

	edm::Service<cond::service::PoolDBOutputService> dbService;
	if (!dbService.isAvailable())
		throw cms::Exception("MVATrainerSave")
			<< "DBService unavailable" << std::endl;

	dbService->createNewIOV<Calibration::MVAComputer>(
		calib.release(), dbService->beginOfTime(),
		dbService->endOfTime(), getRecordName().c_str());

	saved = true;
}

} // namespace PhysicsTools

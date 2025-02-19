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

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooper.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerFileSave.h"

namespace PhysicsTools {

MVATrainerFileSave::MVATrainerFileSave(const edm::ParameterSet &params) :
	trained(params.getUntrackedParameter<bool>("trained", true)),
	saved(false)
{
	std::vector<std::string> names = params.getParameterNames();
	for(std::vector<std::string>::const_iterator iter = names.begin();
	    iter != names.end(); iter++) {
		if (iter->c_str()[0] == '@' || *iter == "trained")
			continue;

		toPut[*iter] = params.getParameter<std::string>(*iter);
	}

}

void MVATrainerFileSave::analyze(const edm::Event& event,
                                 const edm::EventSetup& es)
{
	if (calib.get() || saved)
		return;

	const Calibration::MVAComputerContainer *toPutCalib = getToPut(es);
	if (MVATrainerLooper::isUntrained(toPutCalib))
		return;

	edm::LogInfo("MVATrainerFileSave")
		<< "Got the trained calibration data";

	std::auto_ptr<Calibration::MVAComputerContainer> calib(
					new Calibration::MVAComputerContainer);
	*calib = *toPutCalib;

	this->calib = calib;
}

void MVATrainerFileSave::endJob()
{
	if (!calib.get() || saved)
		return;

	edm::LogInfo("MVATrainerFileSave")
		<< "Saving calibration data into plain MVA files.";

	for(LabelFileMap::const_iterator iter = toPut.begin();
	    iter != toPut.end(); iter++) {
		const Calibration::MVAComputer *calibration =
						&calib->find(iter->first);

		MVAComputer::writeCalibration(iter->second.c_str(),
		                              calibration);
	}

	saved = true;
}

} // namespace PhysicsTools

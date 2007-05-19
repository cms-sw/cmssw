#include <algorithm>
#include <iostream>
#include <string>
#include <memory>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputer.h"

using namespace PhysicsTools;

void GenericMVAJetTagComputer::setEventSetup(const edm::EventSetup &es) const
{
	// Check cacheId of the ES stuff or if m_mvaComputer is null
	// if needed create a new m_mvaComputer with update calib

	// retrieve MVAComputer calibration container
	edm::ESHandle<Calibration::MVAComputerContainer> calibHandle;
	es.get<BTauGenericMVAJetTagComputerRcd>().get(calibHandle);
	const Calibration::MVAComputerContainer *calib = calibHandle.product();

	// check container for changes
	if (m_mvaComputer.get() && calib->changed(m_mvaContainerCacheId)) {
		const Calibration::MVAComputer *computerCalib = 
					&calib->find(m_calibrationLabel);

		// check container content for changes
		if (computerCalib->changed(m_mvaComputerCacheId))
			m_mvaComputer.reset();

		m_mvaContainerCacheId = calib->getCacheId();
	}

	if (!m_mvaComputer.get()) {
		const Calibration::MVAComputer *computerCalib = 
					&calib->find(m_calibrationLabel);

		if (!computerCalib)
			throw cms::Exception("GenericMVAJetTagComputer")
				<< "No training calibration obtained for "
				<< m_calibrationLabel << std::endl;

		// instantiate new MVAComputer with uptodate calibration
		m_mvaComputer = std::auto_ptr<GenericMVAComputer>(
					new GenericMVAComputer(computerCalib));

		m_mvaComputerCacheId = computerCalib->getCacheId();
	}
}

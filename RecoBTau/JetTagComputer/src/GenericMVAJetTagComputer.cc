#include <algorithm>
#include <iostream>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputer.h"

using namespace PhysicsTools;

void GenericMVAJetTagComputer::setEventSetup(const edm::EventSetup &es)
{
	// Check cacheId of the ES stuff or if m_mvaComputer is null
	// if needed create a new m_mvaComputer with update calib
	//

	// m_eventSetup = &es;

	if (m_mvaComputer)
//FIXME		return;
		delete m_mvaComputer;

	edm::ESHandle<Calibration::MVAComputerContainer> calib;
	es.get<BTauGenericMVAJetTagComputerRcd>().get(calib);
	m_mvaComputer = new GenericMVAComputer(
				&calib.product()->find(m_calibrationLabel));
}

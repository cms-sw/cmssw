#include <algorithm>
#include <iostream>
#include <string>
#include <memory>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputer.h"

using namespace reco;
using namespace PhysicsTools;

GenericMVAJetTagComputer::GenericMVAJetTagComputer(const edm::ParameterSet & parameters) :
	m_calibrationLabel(parameters.getParameter<std::string>("calibrationRecord")),
	m_mvaComputerCacheId(Calibration::MVAComputer::CacheId()),
	m_mvaContainerCacheId(Calibration::MVAComputerContainer::CacheId())
{
}

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

float GenericMVAJetTagComputer::discriminator(const BaseTagInfo &baseTag) const
{
	TaggingVariableList variables = baseTag.taggingVariables();
	edm::RefToBase<Jet> jet = baseTag.jet();

	variables.push_back(TaggingVariable(btau::jetPt, jet->pt()));
	variables.push_back(TaggingVariable(btau::jetEta, jet->eta()));

	return m_mvaComputer->eval(variables);
}

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
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"

using namespace reco;
using namespace PhysicsTools;

static std::vector<std::string>
getCalibrationLabels(const edm::ParameterSet &params,
                     std::unique_ptr<TagInfoMVACategorySelector> &selector)
{
	if (params.getParameter<bool>("useCategories")) {
		selector = std::unique_ptr<TagInfoMVACategorySelector>(
				new TagInfoMVACategorySelector(params));

		return selector->getCategoryLabels();
	} else {
		std::string calibrationRecord =
			params.getParameter<std::string>("calibrationRecord");

		std::vector<std::string> calibrationLabels;
		calibrationLabels.push_back(calibrationRecord);
		return calibrationLabels;
	}
}

GenericMVAJetTagComputer::GenericMVAJetTagComputer(
					const edm::ParameterSet &params) :
	computerCache_(getCalibrationLabels(params, categorySelector_)),
        recordLabel_(params.getParameter<std::string>("recordLabel"))
{
}

GenericMVAJetTagComputer::~GenericMVAJetTagComputer()
{
}

void GenericMVAJetTagComputer::initialize(const JetTagComputerRecord & record) {

        const BTauGenericMVAJetTagComputerRcd& dependentRecord = record.getRecord<BTauGenericMVAJetTagComputerRcd>();

	// retrieve MVAComputer calibration container
	edm::ESHandle<Calibration::MVAComputerContainer> calibHandle;
	dependentRecord.get(recordLabel_, calibHandle);
	const Calibration::MVAComputerContainer *calib = calibHandle.product();

	// check for updates
	computerCache_.update(calib);
}

float GenericMVAJetTagComputer::discriminator(const TagInfoHelper &info) const
{
	TaggingVariableList variables = taggingVariables(info);

	// retrieve index of computer in case categories are used
	int index = 0;
	if (categorySelector_.get()) {
		index = categorySelector_->findCategory(variables);
		if (index < 0)
			return -10.0;
	}

	GenericMVAComputer const* computer = computerCache_.getComputer(index);

	if (!computer)
		return -10.0;

	return computer->eval(variables);
}

TaggingVariableList
GenericMVAJetTagComputer::taggingVariables(const BaseTagInfo &baseTag) const
{
	TaggingVariableList variables = baseTag.taggingVariables();

	// add jet pt and jet eta variables (ordering irrelevant)
	edm::RefToBase<Jet> jet = baseTag.jet();
	variables.push_back(TaggingVariable(btau::jetPt, jet->pt()));
	variables.push_back(TaggingVariable(btau::jetEta, jet->eta()));

	return variables;
}

TaggingVariableList
GenericMVAJetTagComputer::taggingVariables(const TagInfoHelper &info) const
{
	return taggingVariables(info.getBase(0));
}

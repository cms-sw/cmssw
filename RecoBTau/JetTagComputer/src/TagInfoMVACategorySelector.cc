#include <ext/functional>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "RecoBTau/JetTagComputer/interface/TagInfoMVACategorySelector.h"

using namespace reco;

TagInfoMVACategorySelector::TagInfoMVACategorySelector(
					const edm::ParameterSet &params)
{
	std::string variableName =
		params.getParameter<std::string>("categoryVariableName");

	categoryVariable = getTaggingVariableName(variableName);
	if (categoryVariable >= btau::lastTaggingVariable)
		throw cms::Exception("TagInfoMVACategorySelector")
			<< "No such tagging variable \""
			<< categoryVariable << "\"." << std::endl;

	categoryLabels = params.getParameter<std::vector<std::string> >(
							"calibrationRecords");
	for(std::vector<std::string>::iterator iter = categoryLabels.begin();
	    iter != categoryLabels.end(); iter++)
		if (*iter == " " || *iter == "-" || *iter == "*")
			*iter = "";
}

TagInfoMVACategorySelector::~TagInfoMVACategorySelector()
{
}

int TagInfoMVACategorySelector::findCategory(
			const TaggingVariableList &taggingVariables) const
{
	int index = (int)taggingVariables.get(categoryVariable, -1);

	if (index < 0 || (unsigned int)index >= categoryLabels.size())
		return -1;

	return index;
}

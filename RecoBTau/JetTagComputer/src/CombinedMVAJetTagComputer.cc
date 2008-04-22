#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/CombinedMVAJetTagComputer.h"

using namespace reco;
using namespace PhysicsTools;

CombinedMVAJetTagComputer::CombinedMVAJetTagComputer(
					const edm::ParameterSet &params) :
	GenericMVAJetTagComputer(params)
{
	std::vector<edm::ParameterSet> computers =
		params.getParameter< std::vector<edm::ParameterSet> >(
							"jetTagComputers");

	for(std::vector<edm::ParameterSet>::const_iterator iter =
		computers.begin(); iter != computers.end(); ++iter) {

		Computer computer;
		computer.name = iter->getParameter<std::string>("jetTagComputer");
		computer.discriminator = iter->getParameter<bool>("discriminator");
		computer.variables = iter->getParameter<bool>("variables");
		computer.computer = 0;

		this->computers.push_back(computer);
	}
}

CombinedMVAJetTagComputer::~CombinedMVAJetTagComputer()
{
}

void CombinedMVAJetTagComputer::setEventSetup(const edm::EventSetup &es) const
{
	std::map<std::string, int> indexMap;
	int index = 0;

	for(std::vector<Computer>::iterator iter = computers.begin();
	    iter != computers.end(); ++iter) {
		if (!iter->computer) {
			edm::ESHandle<JetTagComputer> computer;
			es.get<JetTagComputerRecord>().get(
							iter->name, computer);
			iter->computer = computer.product();

			// finalize the JetTagComputer glue setup
			std::vector<std::string> inputLabels(
					iter->computer->getInputLabels());

			// backward compatible case, use default tagInfo
			if (inputLabels.empty()) {
				std::ostringstream ss;
				ss << "tagInfo" << (iter - computers.begin());
				inputLabels.push_back(ss.str());
			}

			for(std::vector<std::string>::const_iterator label =
							inputLabels.begin();
			    label != inputLabels.end(); ++label) {
				if (indexMap.find(*label) == indexMap.end()) {
					const_cast<CombinedMVAJetTagComputer*>(
						this)->uses(index, *label);
					indexMap[*label] = index;
					iter->indices.push_back(index++);
				} else
					iter->indices.push_back(
							indexMap[*label]);
			}
		}

		iter->computer->setEventSetup(es);
	}

	GenericMVAJetTagComputer::setEventSetup(es);
}

TaggingVariableList
CombinedMVAJetTagComputer::taggingVariables(const TagInfoHelper &info) const
{
	TaggingVariableList vars;

	return vars;
}

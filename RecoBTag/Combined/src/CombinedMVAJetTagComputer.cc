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
#include "RecoBTag/Combined/interface/CombinedMVAJetTagComputer.h"

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

void CombinedMVAJetTagComputer::initialize(const JetTagComputerRecord & record) {

  std::map<std::string, int> indexMap;
  int index = 0;
  int nonameIndex = 0;

  for(std::vector<Computer>::iterator iter = computers.begin();
      iter != computers.end(); ++iter) {

    edm::ESHandle<JetTagComputer> computerHandle;
    record.get(iter->name, computerHandle);
    if (!iter->computer) {
      iter->computer = computerHandle.product();

      // finalize the JetTagComputer glue setup
      std::vector<std::string> inputLabels(iter->computer->getInputLabels());

      // backward compatible case, use default tagInfo
      if (inputLabels.empty()) {
        std::ostringstream ss;
        ss << "tagInfo" << ++nonameIndex;
        inputLabels.push_back(ss.str());
      }
      for(std::vector<std::string>::const_iterator label =
            inputLabels.begin();
          label != inputLabels.end(); ++label) {
        if (indexMap.find(*label) == indexMap.end()) {
          uses(index, *label);
          indexMap[*label] = index;
          iter->indices.push_back(index++);
        } else {
          iter->indices.push_back(indexMap[*label]);
        }
      }
    } else {
      // A sanity check. This should never fail.
      if(iter->computer != computerHandle.product()) {
        throw cms::Exception("LogicError") << "CombinedMVAJetTagComputer::initialize. Pointer to JetTagComputer changed!\n";
      }
    }

    const GenericMVAJetTagComputer *mvaComputer =
      dynamic_cast<const GenericMVAJetTagComputer*>(iter->computer);
    if (iter->variables && !mvaComputer) {
      throw cms::Exception("LogicError")
        << "JetTagComputer \"" << iter->name
        << "\" is not an MVAJetTagCompputer, "
        "but tagging variables have been "
        "requested." << std::endl;
    }
  }
  GenericMVAJetTagComputer::initialize(record);
}

TaggingVariableList
CombinedMVAJetTagComputer::taggingVariables(const TagInfoHelper &info) const
{
	TaggingVariableList vars;
	std::vector<const BaseTagInfo*> tagInfos;

	for(std::vector<Computer>::const_iterator iter = computers.begin();
	    iter != computers.end(); ++iter) {
		if (!iter->computer)
			throw cms::Exception("LogicError")
				<< "JetTagComputer \"" << iter->name
				<< "\" is not available in "
				   "CombinedMVAJetTagComputer::"
				   "taggingVariables()" << std::endl;

		tagInfos.clear();
		for(std::vector<int>::const_iterator i = iter->indices.begin();
		    i != iter->indices.end(); ++i)
			tagInfos.push_back(&info.getBase(*i));

		if (iter->variables) {
			const GenericMVAJetTagComputer *mvaComputer =
				dynamic_cast<const GenericMVAJetTagComputer*>(
							iter->computer);
			vars.insert(mvaComputer->taggingVariables(info));
		}

		if (iter->discriminator)
			vars.insert(btau::algoDiscriminator,
		            (*iter->computer)(TagInfoHelper(tagInfos)));
	}

	return vars;
}

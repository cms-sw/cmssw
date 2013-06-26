#include <algorithm>
#include <iterator>
#include <sstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <string>
#include <set>

#include <xercesc/dom/DOM.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcMultiply : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcMultiply>::Type Registry;

	ProcMultiply(const char *name, const AtomicId *id,
	             MVATrainer *trainer);
	virtual ~ProcMultiply();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

    private:
	typedef std::vector<unsigned int>	Config;
	std::vector<Config>			config;
};

static ProcMultiply::Registry registry("ProcMultiply");

ProcMultiply::ProcMultiply(const char *name, const AtomicId *id,
                           MVATrainer *trainer) :
	TrainProcessor(name, id, trainer)
{
}

ProcMultiply::~ProcMultiply()
{
}

void ProcMultiply::configure(DOMElement *elem)
{
	unsigned int nInputs = getInputs().size();

	for(DOMNode *node = elem->getFirstChild(); node;
	    node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()), "product") != 0)
			throw cms::Exception("ProcMultiply")
				<< "Expected product tag in config section."
				<< std::endl;

		std::string data =
			(const char*)XMLSimpleStr(node->getTextContent());

		Config indices;
		for(size_t pos = 0, next = 0;
		    next != std::string::npos; pos = next + 1) {
			next = data.find('*', pos);

			std::istringstream ss(data.substr(pos, next - pos));
			unsigned int index;
			ss >> index;
			if (ss.bad() || ss.peek() !=
					std::istringstream::traits_type::eof())
				throw cms::Exception("ProcMultiply")
					<< "Expected list of indices separated"
					<< "by asterisks" << std::endl;
			if (index >= nInputs)
				throw cms::Exception("ProcMultiply")
					<< "Variable index " << index
					<< " out of range." << std::endl;
			indices.push_back(index);
		}

		config.push_back(indices);
	}

	trained = true;
}

Calibration::VarProcessor *ProcMultiply::getCalibration() const
{
	Calibration::ProcMultiply *calib = new Calibration::ProcMultiply;
	calib->in = getInputs().size();
	calib->out = config;
	return calib;
}

} // anonymous namespace

#include <algorithm>
#include <iterator>
#include <iostream>
#include <cstring>
#include <vector>
#include <string>

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

class ProcOptional : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcOptional>::Type Registry;

	ProcOptional(const char *name, const AtomicId *id,
	             MVATrainer *trainer);
	virtual ~ProcOptional();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

    private:
	std::vector<double>	neutrals;
};

static ProcOptional::Registry registry("ProcOptional");

ProcOptional::ProcOptional(const char *name, const AtomicId *id,
                             MVATrainer *trainer) :
	TrainProcessor(name, id, trainer)
{
}

ProcOptional::~ProcOptional()
{
}

void ProcOptional::configure(DOMElement *elem)
{
	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                "neutral") != 0)
			throw cms::Exception("ProcOptional")
				<< "Expected neutral tag in config section."
				<< std::endl;
		elem = static_cast<DOMElement*>(node);

		double neutral = 
			XMLDocument::readAttribute<double>(elem, "pos");

		neutrals.push_back(neutral);
	}

	trained = true;

	if (neutrals.size() != getInputs().size())
		throw cms::Exception("ProcOptional")
			<< "Got " << neutrals.size()
			<< " neutral pos values for "
			<< getInputs().size() << " input variables."
			<< std::endl;
}

Calibration::VarProcessor *ProcOptional::getCalibration() const
{
	Calibration::ProcOptional *calib = new Calibration::ProcOptional;

	std::copy(neutrals.begin(), neutrals.end(),
	          std::back_inserter(calib->neutralPos));

	return calib;
}

} // anonymous namespace

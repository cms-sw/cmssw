#include <iostream>
#include <vector>
#include <string>

#include <xercesc/dom/DOM.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcForeach : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcForeach>::Type Registry;

	ProcForeach(const char *name, const AtomicId *id,
	            MVATrainer *trainer);
	virtual ~ProcForeach();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

    private:
	unsigned int	count;
};

static ProcForeach::Registry registry("ProcForeach");

ProcForeach::ProcForeach(const char *name, const AtomicId *id,
                         MVATrainer *trainer) :
	TrainProcessor(name, id, trainer)
{
}

ProcForeach::~ProcForeach()
{
}

void ProcForeach::configure(DOMElement *elem)
{
	DOMNode *node = elem->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node)
		throw cms::Exception("ProcForeach")
			<< "Expected procs tag in config section."
			<< std::endl;

	if (std::strcmp(XMLSimpleStr(node->getNodeName()), "procs") != 0)
		throw cms::Exception("ProcForeach")
				<< "Expected procs tag in config section."
				<< std::endl;

	elem = static_cast<DOMElement*>(node);

	count = XMLDocument::readAttribute<unsigned int>(elem, "next");

	node = node->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcForeach")
			<< "Superfluous tags in config section."
			<< std::endl;

	trained = true;
}

Calibration::VarProcessor *ProcForeach::getCalibration() const
{
	Calibration::ProcForeach *calib = new Calibration::ProcForeach;
	calib->nProcs = count;
	return calib;
}

} // anonymous namespace

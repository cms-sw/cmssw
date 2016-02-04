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

class ProcSplitter : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcSplitter>::Type Registry;

	ProcSplitter(const char *name, const AtomicId *id,
	             MVATrainer *trainer);
	virtual ~ProcSplitter();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

    private:
	unsigned int	count;
};

static ProcSplitter::Registry registry("ProcSplitter");

ProcSplitter::ProcSplitter(const char *name, const AtomicId *id,
                           MVATrainer *trainer) :
	TrainProcessor(name, id, trainer)
{
}

ProcSplitter::~ProcSplitter()
{
}

void ProcSplitter::configure(DOMElement *elem)
{
	DOMNode *node = elem->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node ||
	    std::strcmp(XMLSimpleStr(node->getNodeName()), "select") != 0)
		throw cms::Exception("ProcSplitter")
			<< "Expected select tag in config section."
			<< std::endl;

	elem = static_cast<DOMElement*>(node);

	count = XMLDocument::readAttribute<unsigned int>(elem, "first");

	node = node->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcSplitter")
			<< "Superfluous tags in config section."
			<< std::endl;

	trained = true;
}

Calibration::VarProcessor *ProcSplitter::getCalibration() const
{
	Calibration::ProcSplitter *calib = new Calibration::ProcSplitter;
	calib->nFirst = count;
	return calib;
}

} // anonymous namespace

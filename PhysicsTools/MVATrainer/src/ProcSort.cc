#include <algorithm>
#include <iterator>
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

struct Range {
	bool	hasMin, hasMax;
	double	min, max;
};

struct Box {
	std::vector<Range>	ranges;
	int			group;
};

class ProcSort : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcSort>::Type Registry;

	ProcSort(const char *name, const AtomicId *id, MVATrainer *trainer);
	virtual ~ProcSort();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

    private:
	unsigned int	sortByIndex;
	bool		descending;
};

static ProcSort::Registry registry("ProcSort");

ProcSort::ProcSort(const char *name, const AtomicId *id, MVATrainer *trainer) :
	TrainProcessor(name, id, trainer)
{
}

ProcSort::~ProcSort()
{
}

void ProcSort::configure(DOMElement *elem)
{
	DOMNode *node = elem->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node ||
	    std::strcmp(XMLSimpleStr(node->getNodeName()), "key") != 0)
		throw cms::Exception("ProcSort")
			<< "Expected key tag in config section."
			<< std::endl;

	elem = static_cast<DOMElement*>(node);

	sortByIndex = XMLDocument::readAttribute<unsigned int>(elem, "index");
	descending = XMLDocument::readAttribute<bool>(elem, "descending",
	                                              false);

	if (sortByIndex >= getInputs().size())
		throw cms::Exception("ProcSort")
			<< "Key index out of bounds." << std::endl;

	node = node->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcSort")
			<< "Superfluous tags in config section."
			<< std::endl;

	trained = true;
}

Calibration::VarProcessor *ProcSort::getCalibration() const
{
	Calibration::ProcSort *calib = new Calibration::ProcSort;

	calib->sortByIndex = sortByIndex;
	calib->descending = descending;

	return calib;
}

} // anonymous namespace

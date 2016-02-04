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

class ProcCategory : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcCategory>::Type Registry;

	ProcCategory(const char *name, const AtomicId *id,
	             MVATrainer *trainer);
	virtual ~ProcCategory();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

    private:
	typedef Calibration::ProcCategory::BinLimits BinLimits;

	std::vector<int>	matrix;
	std::vector<BinLimits>	limits;
};

static ProcCategory::Registry registry("ProcCategory");

ProcCategory::ProcCategory(const char *name, const AtomicId *id,
                           MVATrainer *trainer) :
	TrainProcessor(name, id, trainer)
{
}

ProcCategory::~ProcCategory()
{
}

static void
fillRange(unsigned int n, int *matrix, unsigned int off,
          const unsigned int *strides,
          const std::pair<unsigned int, unsigned int> *ranges, int value)
{
	for(unsigned int i = ranges->first; i < ranges->second; i++) {
		if (n > 1)
			fillRange(n - 1, matrix, off + *strides * i,
			          strides + 1, ranges + 1, value);
		else
			matrix[off + i] = value;
	}
}

void ProcCategory::configure(DOMElement *elem)
{
	unsigned int n = getInputs().size();
	std::vector<Box> boxes;
	std::vector< std::set<double> > sLimits(n);

	int group = 0;
	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                "group") != 0)
			throw cms::Exception("ProcCategory")
				<< "Expected group tag in config section."
				<< std::endl;

		for(DOMNode *subNode = node->getFirstChild();
		    subNode; subNode = subNode->getNextSibling()) {
			if (subNode->getNodeType() != DOMNode::ELEMENT_NODE)
				continue;

			if (std::strcmp(XMLSimpleStr(subNode->getNodeName()),
			                "box") != 0)
				throw cms::Exception("ProcCategory")
					<< "Expected box tag in config section."
					<< std::endl;

			Box box;
			box.group = group;

			unsigned int i = 0;
			for(DOMNode *rangeNode = subNode->getFirstChild();
			    rangeNode; rangeNode = rangeNode->getNextSibling()) {
				if (rangeNode->getNodeType() != DOMNode::ELEMENT_NODE)
					continue;

				if (std::strcmp(XMLSimpleStr(rangeNode->getNodeName()),
				                "range") != 0)
					throw cms::Exception("ProcCategory")
						<< "Expected range tag in config section."
						<< std::endl;

				if (i >= n)
					throw cms::Exception("ProcCategory")
						<< "Number of ranges exceeded."
						<< std::endl;

				elem = static_cast<DOMElement*>(rangeNode);

				Range range;
				if (XMLDocument::hasAttribute(elem, "min")) {
					range.min = XMLDocument::readAttribute<double>(elem, "min");
					range.hasMin = true;
					sLimits[i].insert(range.min);
				} else
					range.hasMin = false;

				if (XMLDocument::hasAttribute(elem, "max")) {
					range.max = XMLDocument::readAttribute<double>(elem, "max");
					range.hasMax = true;
					sLimits[i].insert(range.max);
				} else
					range.hasMax = false;

				box.ranges.push_back(range);
				i++;
			}
			if (i < n)
				throw cms::Exception("ProcCategory")
					<< "Not enough ranges." << std::endl;

			boxes.push_back(box);
		}

		group++;
	}

	limits.clear();
	limits.resize(n);
	for(unsigned int i = 0; i < n; i++)
		std::copy(sLimits[i].begin(), sLimits[i].end(),
		          std::back_inserter(limits[i]));
	sLimits.clear();

	unsigned int total = 1;
	std::vector<unsigned int> strides;
	for(unsigned int i = n; i > 0; i--) {
		strides.push_back(total);
		total *= limits[i - 1].size() + 1;
	}
	std::reverse(strides.begin(), strides.end());
	matrix.clear();
	matrix.resize(total, -1);

	std::vector< std::pair<unsigned int, unsigned int> > ranges(n);

	for(std::vector<Box>::reverse_iterator iter = boxes.rbegin();
	    iter != boxes.rend(); iter++) {
		for(unsigned int i = 0; i < n; i++) {
			const Range &range = iter->ranges[i];

			unsigned int minIdx;
			if (range.hasMin) {
				BinLimits::const_iterator pos =
					std::find(limits[i].begin(),
					          limits[i].end(), range.min);
				if (pos == limits[i].end())
					throw cms::Exception("ProcCategory")
						<< "Fatal: min limit not found"
						<< std::endl;
				minIdx = (pos - limits[i].begin()) + 1;
			} else
				minIdx = 0;

			unsigned int maxIdx;
			if (range.hasMax) {
				BinLimits::const_iterator pos =
					std::find(limits[i].begin(),
					          limits[i].end(), range.max);
				if (pos == limits[i].end())
					throw cms::Exception("ProcCategory")
						<< "Fatal: max limit not found"
						<< std::endl;
				maxIdx = (pos - limits[i].begin()) + 1;
			} else
				maxIdx = limits[i].size() + 1;

			ranges[i] = std::make_pair(minIdx, maxIdx);
		}

		fillRange(n, &matrix.front(), 0,
		          &strides.front(), &ranges.front(), iter->group);
	}

	trained = true;
}

Calibration::VarProcessor *ProcCategory::getCalibration() const
{
	Calibration::ProcCategory *calib = new Calibration::ProcCategory;

	calib->variableBinLimits = limits;
	calib->categoryMapping = matrix;

	return calib;
}

} // anonymous namespace

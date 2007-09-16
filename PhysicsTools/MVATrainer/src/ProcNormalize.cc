#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <string>
#include <memory>

#include <xercesc/dom/DOM.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/XMLUniStr.h"
#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcNormalize : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcNormalize>::Type Registry;

	ProcNormalize(const char *name, const AtomicId *id,
	              MVATrainer *trainer);
	virtual ~ProcNormalize();

	virtual Variable::Flags getDefaultFlags() const
	{ return Variable::FLAG_ALL; }

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

	virtual void trainBegin();
	virtual void trainData(const std::vector<double> *values,
	                       bool target, double weight);
	virtual void trainEnd();

	virtual bool load();
	virtual void save();
	
    private:
	enum Iteration {
		ITER_EMPTY,
		ITER_RANGE,
		ITER_FILL,
		ITER_DONE
	};

	struct PDF {
		operator Calibration::Histogram() const
		{
			Calibration::Histogram histo(distr.size(), range);
			for(unsigned int i = 0; i < distr.size(); i++)
				histo.setBinContent(i + 1, distr[i]);
			histo.normalize();
			return histo;
		}

		unsigned int			smooth;
		std::vector<float>		distr;
		Calibration::Histogram::Range	range;
		Iteration			iteration;
	};

	std::vector<PDF> pdfs;
};

static ProcNormalize::Registry registry("ProcNormalize");

ProcNormalize::ProcNormalize(const char *name, const AtomicId *id,
                             MVATrainer *trainer) :
	TrainProcessor(name, id, trainer)
{
}

ProcNormalize::~ProcNormalize()
{
}

void ProcNormalize::configure(DOMElement *elem)
{
	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()), "pdf") != 0)
			throw cms::Exception("ProcNormalize")
				<< "Expected pdf tag in config section."
				<< std::endl;
		elem = static_cast<DOMElement*>(node);

		PDF pdf;

		pdf.distr.resize(XMLDocument::readAttribute<unsigned int>(
							elem, "size", 100));

		pdf.smooth = XMLDocument::readAttribute<unsigned int>(
							elem, "smooth", 40);

		try {
			pdf.range.min = XMLDocument::readAttribute<float>(
								elem, "lower");
			pdf.range.max = XMLDocument::readAttribute<float>(
								elem, "upper");
			pdf.iteration = ITER_FILL;
		} catch(...) {
			pdf.iteration = ITER_EMPTY;
		}

		pdfs.push_back(pdf);
	}

	if (pdfs.size() != getInputs().size())
		throw cms::Exception("ProcNormalize")
			<< "Got " << pdfs.size() << " pdf configs for "
			<< getInputs().size() << " input varibles."
			<< std::endl;
}

Calibration::VarProcessor *ProcNormalize::getCalibration() const
{
	Calibration::ProcNormalize *calib = new Calibration::ProcNormalize;
	std::copy(pdfs.begin(), pdfs.end(), std::back_inserter(calib->distr));
	calib->nCategories = 0;
	return calib;
}

void ProcNormalize::trainBegin()
{
}

void ProcNormalize::trainData(const std::vector<double> *values,
                              bool target, double weight)
{
	for(std::vector<PDF>::iterator iter = pdfs.begin();
	    iter != pdfs.end(); iter++, values++) {
		switch(iter->iteration) {
		    case ITER_EMPTY:
			for(std::vector<double>::const_iterator value =
							values->begin();
				value != values->end(); value++) {
				iter->range.min = iter->range.max = *value;
				iter->iteration = ITER_RANGE;
				break;
			}
		    case ITER_RANGE:
			for(std::vector<double>::const_iterator value =
							values->begin();
				value != values->end(); value++) {
				iter->range.min = std::min(iter->range.min,
				                           (float)*value);
				iter->range.max = std::max(iter->range.max,
				                           (float)*value);
			}
			continue;
		    case ITER_FILL:
			break;
		    default:
			continue;
		}

		unsigned int n = iter->distr.size() - 1;
		double mult = 1.0 / iter->range.width();

		for(std::vector<double>::const_iterator value =
			values->begin(); value != values->end(); value++) {
			double x = (*value - iter->range.min) * mult;
			if (x < 0.0)
				x = 0.0;
			else if (x >= 1.0)
				x = 1.0;

			iter->distr[(unsigned int)(x * n + 0.5)] += weight;
		}
	}
}

static void smoothArray(unsigned int n, float *values, unsigned int nTimes)
{
	for(unsigned int iter = 0; iter < nTimes; iter++) {
		float hold = n > 0 ? values[0] : 0.0;
		for(unsigned int i = 0; i < n; i++) {
			float delta = hold * 0.1;
			float rem = 0.0;
			if (i > 0) {
				values[i - 1] += delta;
				rem -= delta;
			}
			if (i < n - 1) {
				hold = values[i + 1];
				values[i + 1] += delta;
				rem -= delta;
			}
			values[i] += rem;
		}
	}
}

void ProcNormalize::trainEnd()
{
	bool done = true;
	for(std::vector<PDF>::iterator iter = pdfs.begin();
	    iter != pdfs.end(); iter++) {
		switch(iter->iteration) {
		    case ITER_EMPTY:
		    case ITER_RANGE:
			iter->iteration = ITER_FILL;
			done = false;
			break;
		    case ITER_FILL:
			iter->distr.front() *= 2;
			iter->distr.back() *= 2;
			smoothArray(iter->distr.size(),
			            &iter->distr.front(),
			            iter->smooth);

			iter->iteration = ITER_DONE;
			break;
		    default:
			/* shut up */;
		}
	}

	if (done)
		trained = true;
}

bool ProcNormalize::load()
{
	std::auto_ptr<XMLDocument> xml;

	try {
		xml = std::auto_ptr<XMLDocument>(new XMLDocument(
				trainer->trainFileName(this, "xml")));
	} catch(...) {
		return false;
	}

	DOMElement *elem = xml->getRootNode();
	if (std::strcmp(XMLSimpleStr(elem->getNodeName()),
	                             "ProcNormalize") != 0)
		throw cms::Exception("ProcNormalize")
			<< "XML training data file has bad root node."
			<< std::endl;

	std::vector<PDF>::iterator cur = pdfs.begin();

	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (cur == pdfs.end())
			throw cms::Exception("ProcNormalize")
				<< "Superfluous PDF in train data."
				<< std::endl;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()), "pdf") != 0)
			throw cms::Exception("ProcNormalize")
				<< "Expected pdf tag in train file."
				<< std::endl;
		elem = static_cast<DOMElement*>(node);

		PDF pdf;

		pdf.range.min =
			XMLDocument::readAttribute<float>(elem, "lower");
		pdf.range.max =
			XMLDocument::readAttribute<float>(elem, "upper");
		pdf.iteration = ITER_DONE;

		for(DOMNode *subNode = elem->getFirstChild();
		    subNode; subNode = subNode->getNextSibling()) {
			if (subNode->getNodeType() != DOMNode::ELEMENT_NODE)
				continue;

			if (std::strcmp(XMLSimpleStr(subNode->getNodeName()),
			                             "value") != 0)
				throw cms::Exception("ProcNormalize")
					<< "Expected value tag in train file."
					<< std::endl;

			elem = static_cast<DOMElement*>(node);

			pdf.distr.push_back(
				XMLDocument::readContent<float>(subNode));
		}

		*cur++ = pdf;
	}

	if (cur != pdfs.end())
		throw cms::Exception("ProcNormalize")
			<< "Missing PDF in train data." << std::endl;

	trained = true;
	return true;
}

void ProcNormalize::save()
{
	XMLDocument xml(trainer->trainFileName(this, "xml"), true);
	DOMDocument *doc = xml.createDocument("ProcNormalize");

	for(std::vector<PDF>::const_iterator iter = pdfs.begin();
	    iter != pdfs.end(); iter++) {
		DOMElement *elem = doc->createElement(XMLUniStr("pdf"));
		xml.getRootNode()->appendChild(elem);

		XMLDocument::writeAttribute(elem, "lower", iter->range.min);
		XMLDocument::writeAttribute(elem, "upper", iter->range.max);

		for(std::vector<float>::const_iterator iter2 =
		    iter->distr.begin(); iter2 != iter->distr.end(); iter2++) {
			DOMElement *value =
					doc->createElement(XMLUniStr("value"));
			elem->appendChild(value);	

			XMLDocument::writeContent<float>(value, doc, *iter2);
		}
	}
}

} // anonymous namespace

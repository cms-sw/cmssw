#include <algorithm>
#include <iostream>
#include <numeric>
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

class ProcLikelihood : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcLikelihood>::Type Registry;

	ProcLikelihood(const char *name, const AtomicId *id,
	               MVATrainer *trainer);
	virtual ~ProcLikelihood();

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

	struct PDF {
		std::vector<float>		distr;
		Calibration::Histogram::Range	range;
	};

    private:
	enum Iteration {
		ITER_EMPTY,
		ITER_RANGE,
		ITER_FILL,
		ITER_DONE
	};

	struct SigBkg {
		PDF		signal;
		PDF		background;
		unsigned int	smooth;
		Iteration	iteration;
	};

	std::vector<SigBkg> pdfs;
};

static ProcLikelihood::Registry registry("ProcLikelihood");

ProcLikelihood::ProcLikelihood(const char *name, const AtomicId *id,
                               MVATrainer *trainer) :
	TrainProcessor(name, id, trainer)
{
}

ProcLikelihood::~ProcLikelihood()
{
}

void ProcLikelihood::configure(DOMElement *elem)
{
	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                "sigbkg") != 0)
			throw cms::Exception("ProcLikelihood")
				<< "Expected sigbkg tag in config section."
				<< std::endl;
		elem = static_cast<DOMElement*>(node);

		SigBkg pdf;

		unsigned int size = XMLDocument::readAttribute<unsigned int>(
							elem, "size", 50);
		pdf.signal.distr.resize(size);
		pdf.background.distr.resize(size);

		pdf.smooth = XMLDocument::readAttribute<unsigned int>(
							elem, "smooth", 0);

		try {
			pdf.signal.range.min =
				XMLDocument::readAttribute<float>(
								elem, "lower");
			pdf.signal.range.max =
				XMLDocument::readAttribute<float>(
								elem, "upper");
			pdf.background.range = pdf.signal.range;
			pdf.iteration = ITER_FILL;
		} catch(...) {
			pdf.iteration = ITER_EMPTY;
		}

		pdfs.push_back(pdf);
	}

	if (pdfs.size() != getInputs().size())
		throw cms::Exception("ProcLikelihood")
			<< "Got " << pdfs.size() << " pdf configs for "
			<< getInputs().size() << " input varibles."
			<< std::endl;
}

Calibration::VarProcessor *ProcLikelihood::getCalibration() const
{
	Calibration::ProcLikelihood *calib = new Calibration::ProcLikelihood;

	for(std::vector<SigBkg>::const_iterator iter = pdfs.begin();
	    iter != pdfs.end(); iter++) {
		Calibration::ProcLikelihood::SigBkg pdf;

		pdf.signal =
			Calibration::Histogram(iter->signal.distr.size(),
			                       iter->signal.range);
		double factor = std::accumulate(iter->signal.distr.begin(),
			                        iter->signal.distr.end(), 0.0);
		factor = 1.0 / factor;
		std::transform(iter->signal.distr.begin(),
		               iter->signal.distr.end(),
		               pdf.signal.getValueArray().begin() + 1,
		               std::bind1st(std::multiplies<float>(), factor));
		pdf.signal.normalize();

		pdf.background =
			Calibration::Histogram(iter->background.distr.size(),
			                       iter->background.range);
		factor = std::accumulate(iter->background.distr.begin(),
		                         iter->background.distr.end(), 0.0);
		factor = 1.0 / factor;
		std::transform(iter->background.distr.begin(),
		               iter->background.distr.end(),
		               pdf.background.getValueArray().begin() + 1,
		               std::bind1st(std::multiplies<float>(), factor));
		pdf.background.normalize();

		pdf.useSplines = true;

		calib->pdfs.push_back(pdf);
	}

	calib->nCategories = 0;
	calib->bias = 1.0;

	return calib;
}

void ProcLikelihood::trainBegin()
{
}

void ProcLikelihood::trainData(const std::vector<double> *values,
                               bool target, double weight)
{
	for(std::vector<SigBkg>::iterator iter = pdfs.begin();
	    iter != pdfs.end(); iter++, values++) {
		switch(iter->iteration) {
		    case ITER_EMPTY:
			for(std::vector<double>::const_iterator value =
							values->begin();
				value != values->end(); value++) {
				iter->signal.range.min =
					iter->signal.range.max = *value;
				iter->iteration = ITER_RANGE;
				break;
			}
		    case ITER_RANGE:
			for(std::vector<double>::const_iterator value =
							values->begin();
				value != values->end(); value++) {
				iter->signal.range.min =
					std::min(iter->signal.range.min,
					         (float)*value);
				iter->signal.range.max =
					std::max(iter->signal.range.max,
					         (float)*value);
			}
			continue;
		    case ITER_FILL:
			break;
		    default:
			continue;
		}

		PDF &pdf = target ? iter->signal : iter->background;
		unsigned int n = pdf.distr.size() - 1;
		double mult = 1.0 / pdf.range.width();
 
		for(std::vector<double>::const_iterator value =
			values->begin(); value != values->end(); value++) {
			double x = (*value - pdf.range.min) * mult;
			if (x < 0.0)
				x = 0.0;
			else if (x >= 1.0)
				x = 1.0;

			pdf.distr[(unsigned int)(x * n + 0.5)] += weight;
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

void ProcLikelihood::trainEnd()
{
	bool done = true;
	for(std::vector<SigBkg>::iterator iter = pdfs.begin();
	    iter != pdfs.end(); iter++) {
		switch(iter->iteration) {
		    case ITER_EMPTY:
		    case ITER_RANGE:
			iter->background.range = iter->signal.range;
			iter->iteration = ITER_FILL;
			done = false;
			break;
		    case ITER_FILL:
			iter->signal.distr.front() *= 2;
			iter->signal.distr.back() *= 2;
			smoothArray(iter->signal.distr.size(),
			            &iter->signal.distr.front(),
			            iter->smooth);

			iter->background.distr.front() *= 2;
			iter->background.distr.back() *= 2;
			smoothArray(iter->background.distr.size(),
			            &iter->background.distr.front(),
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

static void xmlParsePDF(ProcLikelihood::PDF &pdf, DOMElement *elem)
{
	if (!elem ||
	    std::strcmp(XMLSimpleStr(elem->getNodeName()), "pdf") != 0)
		throw cms::Exception("ProcLikelihood")
			<< "Expected pdf tag in sigbkg train data."
			<< std::endl;

	pdf.range.min = XMLDocument::readAttribute<float>(elem, "lower");
	pdf.range.max = XMLDocument::readAttribute<float>(elem, "upper");

	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                             "value") != 0)
			throw cms::Exception("ProcLikelihood")
				<< "Expected value tag in train file."
				<< std::endl;

		pdf.distr.push_back(XMLDocument::readContent<float>(node));
	}
}

bool ProcLikelihood::load()
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
	                             "ProcLikelihood") != 0)
		throw cms::Exception("ProcLikelihood")
			<< "XML training data file has bad root node."
			<< std::endl;

	std::vector<SigBkg>::iterator cur = pdfs.begin();

	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (cur == pdfs.end())
			throw cms::Exception("ProcLikelihood")
				<< "Superfluous SigBkg in train data."
				<< std::endl;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                "sigbkg") != 0)
			throw cms::Exception("ProcLikelihood")
				<< "Expected sigbkg tag in train file."
				<< std::endl;
		elem = static_cast<DOMElement*>(node);

		for(node = elem->getFirstChild();
		    node && node->getNodeType() != DOMNode::ELEMENT_NODE;
		    node = node->getNextSibling());
		DOMElement *elemSig =
				node ? static_cast<DOMElement*>(node) : 0;

		for(node = node->getNextSibling();
		    node && node->getNodeType() != DOMNode::ELEMENT_NODE;
		    node = node->getNextSibling());
		while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
			node = node->getNextSibling();
		DOMElement *elemBkg =
				node ? static_cast<DOMElement*>(node) : 0;

		for(node = node->getNextSibling();
		    node && node->getNodeType() != DOMNode::ELEMENT_NODE;
		    node = node->getNextSibling());
		if (node)
			throw cms::Exception("ProcLikelihood")
				<< "Superfluous tags in sigbkg train data."
				<< std::endl;

		SigBkg pdf;

		xmlParsePDF(pdf.signal, elemSig);
		xmlParsePDF(pdf.background, elemBkg);

		pdf.iteration = ITER_DONE;

		*cur++ = pdf;
		node = elem;
	}

	if (cur != pdfs.end())
		throw cms::Exception("ProcLikelihood")
			<< "Missing SigBkg in train data." << std::endl;

	trained = true;
	return true;
}

static DOMElement *xmlStorePDF(DOMDocument *doc,
                               const ProcLikelihood::PDF &pdf)
{
	DOMElement *elem = doc->createElement(XMLUniStr("pdf"));

	XMLDocument::writeAttribute(elem, "lower", pdf.range.min);
	XMLDocument::writeAttribute(elem, "upper", pdf.range.max);

	for(std::vector<float>::const_iterator iter =
	    pdf.distr.begin(); iter != pdf.distr.end(); iter++) {
		DOMElement *value = doc->createElement(XMLUniStr("value"));
		elem->appendChild(value);	

		XMLDocument::writeContent<float>(value, doc, *iter);
	}

	return elem;
}

void ProcLikelihood::save()
{
	XMLDocument xml(trainer->trainFileName(this, "xml"), true);
	DOMDocument *doc = xml.createDocument("ProcLikelihood");

	for(std::vector<SigBkg>::const_iterator iter = pdfs.begin();
	    iter != pdfs.end(); iter++) {
		DOMElement *elem = doc->createElement(XMLUniStr("sigbkg"));
		xml.getRootNode()->appendChild(elem);

		elem->appendChild(xmlStorePDF(doc, iter->signal));
		elem->appendChild(xmlStorePDF(doc, iter->background));
	}
}

} // anonymous namespace

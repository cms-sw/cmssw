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

    private:
	enum Iteration {
		ITER_EMPTY,
		ITER_RANGE,
		ITER_FILL,
		ITER_DONE
	};

	struct SigBkg : public Calibration::ProcLikelihood::SigBkg {
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
			pdf.signal.range.first =
					XMLDocument::readAttribute<double>(
								elem, "lower");
			pdf.signal.range.second =
					XMLDocument::readAttribute<double>(
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

		pdf.signal.range = iter->signal.range;
		double factor = 0.0;
		for(std::vector<double>::const_iterator iter2 =
						iter->signal.distr.begin();
		    iter2 != iter->signal.distr.end(); iter2++)
			factor += *iter2;
		factor = 1.0 / factor;
		pdf.signal.distr.resize(iter->signal.distr.size());
		std::vector<double>::iterator insert =
						pdf.signal.distr.begin();
		for(std::vector<double>::const_iterator iter2 =
						iter->signal.distr.begin();
		    iter2 != iter->signal.distr.end(); iter2++)
			*insert++ = *iter2 * factor;

		pdf.background.range = iter->background.range;
		factor = 0.0;
		for(std::vector<double>::const_iterator iter2 =
						iter->background.distr.begin();
		    iter2 != iter->background.distr.end(); iter2++)
			factor += *iter2;
		factor = 1.0 / factor;
		pdf.background.distr.resize(iter->background.distr.size());
		insert = pdf.background.distr.begin();
		for(std::vector<double>::const_iterator iter2 =
						iter->background.distr.begin();
		    iter2 != iter->background.distr.end(); iter2++)
			*insert++ = *iter2 * factor;

		calib->pdfs.push_back(pdf);
	}

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
				iter->signal.range.first =
					iter->signal.range.second = *value;
				iter->iteration = ITER_RANGE;
				break;
			}
		    case ITER_RANGE:
			for(std::vector<double>::const_iterator value =
							values->begin();
				value != values->end(); value++) {
				iter->signal.range.first =
					std::min(iter->signal.range.first,
					         *value);
				iter->signal.range.second =
					std::max(iter->signal.range.second,
					         *value);
			}
			continue;
		    case ITER_FILL:
			break;
		    default:
			continue;
		}

		Calibration::PDF &pdf = target ? iter->signal
		                               : iter->background;
		unsigned int n = pdf.distr.size() - 1;
		double mult = 1.0 / (pdf.range.second - pdf.range.first);
 
		for(std::vector<double>::const_iterator value =
			values->begin(); value != values->end(); value++) {
			double x = (*value - pdf.range.first) * mult;
			if (x < 0.0)
				x = 0.0;
			else if (x >= 1.0)
				x = 1.0;

			pdf.distr[(unsigned int)(x * n + 0.5)] += weight;
		}
	}
}

static void smoothArray(unsigned int n, double *values, unsigned int nTimes)
{
	for(unsigned int iter = 0; iter < nTimes; iter++) {
		double hold = n > 0 ? values[0] : 0.0;
		for(unsigned int i = 0; i < n; i++) {
			double delta = hold * 0.1;
			double rem = 0.0;
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

static void xmlParsePDF(Calibration::PDF &pdf, DOMElement *elem) {
	if (!elem ||
	    std::strcmp(XMLSimpleStr(elem->getNodeName()), "pdf") != 0)
		throw cms::Exception("ProcLikelihood")
			<< "Expected pdf tag in sigbkg train data."
			<< std::endl;

	pdf.range.first = XMLDocument::readAttribute<double>(elem, "lower");
	pdf.range.second = XMLDocument::readAttribute<double>(elem, "upper");

	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                             "value") != 0)
			throw cms::Exception("ProcLikelihood")
				<< "Expected value tag in train file."
				<< std::endl;

		pdf.distr.push_back(XMLDocument::readContent<double>(node));
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

static DOMElement *xmlStorePDF(DOMDocument *doc, const Calibration::PDF &pdf)
{
	DOMElement *elem = doc->createElement(XMLUniStr("pdf"));

	XMLDocument::writeAttribute(elem, "lower", pdf.range.first);
	XMLDocument::writeAttribute(elem, "upper", pdf.range.second);

	for(std::vector<double>::const_iterator iter =
	    pdf.distr.begin(); iter != pdf.distr.end(); iter++) {
		DOMElement *value = doc->createElement(XMLUniStr("value"));
		elem->appendChild(value);	

		XMLDocument::writeContent<double>(value, doc, *iter);
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

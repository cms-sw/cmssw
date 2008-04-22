#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <map>

#include <xercesc/dom/DOM.hpp>

#include <TH1.h>

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
		operator Calibration::HistogramF() const
		{
			Calibration::HistogramF histo(distr.size(), range);
			for(unsigned int i = 0; i < distr.size(); i++)
				histo.setBinContent(i + 1, distr[i]);
			return histo;
		}

		unsigned int			smooth;
		std::vector<double>		distr;
		Calibration::HistogramD::Range	range;
		Iteration			iteration;
		bool				fillSignal;
		bool				fillBackground;
	};

	std::vector<PDF>	pdfs;
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

		pdf.fillSignal =
			XMLDocument::readAttribute<bool>(elem, "signal", true);
		pdf.fillBackground =
			XMLDocument::readAttribute<bool>(elem, "background", true);

		if (!pdf.fillSignal && !pdf.fillBackground)
			throw cms::Exception("ProcNormalize")
				<< "Filling neither background nor signal "
				   "in config." << std::endl;

		if (XMLDocument::hasAttribute(elem, "lower") &&
		    XMLDocument::hasAttribute(elem, "upper")) {
			pdf.range.min = XMLDocument::readAttribute<double>(
								elem, "lower");
			pdf.range.max = XMLDocument::readAttribute<double>(
								elem, "upper");
			pdf.iteration = ITER_FILL;
		} else
			pdf.iteration = ITER_EMPTY;

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
	calib->categoryIdx = -1;
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
				                           *value);
				iter->range.max = std::max(iter->range.max,
				                           *value);
			}
			continue;
		    case ITER_FILL:
			break;
		    default:
			continue;
		}

		if (!(target ? iter->fillSignal : iter->fillBackground))
			continue;

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

	if (done && monitoring) {
		std::vector<SourceVariable*> inputs = getInputs().get();
		for(std::vector<PDF>::iterator iter = pdfs.begin();
		    iter != pdfs.end(); iter++) {
			SourceVariable *var = inputs[iter - pdfs.begin()];
			std::string name =
				(const char*)var->getSource()->getName()
				+ std::string("_")
				+ (const char*)var->getName();
			unsigned int n = iter->distr.size() - 1;
			double min = iter->range.min -
			             0.5 * iter->range.width() / n;
			double max = iter->range.max +
			             0.5 * iter->range.width() / n;
			TH1F *histo = monitoring->book<TH1F>(name + "_pdf",
				name.c_str(), name.c_str(), n + 1, min, max);
			for(unsigned int i = 0; i < n; i++)
				histo->SetBinContent(i + 1, iter->distr[i]);
		}
	}
}

bool ProcNormalize::load()
{
	std::string filename = trainer->trainFileName(this, "xml");
	if (!exists(filename))
		return false;

	XMLDocument xml(filename);
	DOMElement *elem = xml.getRootNode();
	if (std::strcmp(XMLSimpleStr(elem->getNodeName()),
	                             "ProcNormalize") != 0)
		throw cms::Exception("ProcNormalize")
			<< "XML training data file has bad root node."
			<< std::endl;

	unsigned int version = XMLDocument::readAttribute<unsigned int>(
							elem, "version", 1);

	if (version < 1 || version > 2)
		throw cms::Exception("ProcNormalize")
			<< "Unsupported version " << version
			<< "in train file." << std::endl;

	typedef std::pair<AtomicId, AtomicId> Id;
	std::map<Id, PDF*> pdfMap;

	for(std::vector<PDF>::iterator iter = pdfs.begin();
	    iter != pdfs.end(); ++iter) {
		PDF *ptr = &*iter;
		unsigned int i = iter - pdfs.begin();
		const SourceVariable *var = getInputs().get()[i];
		Id id(var->getSource()->getName(), var->getName());

		pdfMap[id] = ptr;
	}

	std::vector<PDF>::iterator cur = pdfs.begin();

	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()), "pdf") != 0)
			throw cms::Exception("ProcNormalize")
				<< "Expected pdf tag in train file."
				<< std::endl;
		elem = static_cast<DOMElement*>(node);

		PDF *pdf = 0;
		switch(version) {
		    case 1:
			if (cur == pdfs.end())
				throw cms::Exception("ProcNormalize")
					<< "Superfluous PDF in train data."
					<< std::endl;
			pdf = &*cur++;
			break;
		    case 2: {
			Id id(XMLDocument::readAttribute<std::string>(
			      				elem, "source"),
			      XMLDocument::readAttribute<std::string>(
			      				elem, "name"));
			std::map<Id, PDF*>::const_iterator pos =
							pdfMap.find(id);
			if (pos == pdfMap.end())
				continue;
			else
				pdf = pos->second;
		    }	break;
		}

		pdf->range.min =
			XMLDocument::readAttribute<double>(elem, "lower");
		pdf->range.max =
			XMLDocument::readAttribute<double>(elem, "upper");
		pdf->iteration = ITER_DONE;
		pdf->distr.clear();

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

			pdf->distr.push_back(
				XMLDocument::readContent<double>(subNode));
		}
	}

	if (version == 1 && cur != pdfs.end())
		throw cms::Exception("ProcNormalize")
			<< "Missing PDF in train data." << std::endl;

	trained = true;
	for(std::vector<PDF>::const_iterator iter = pdfs.begin();
	    iter != pdfs.end(); ++iter) {
		if (iter->iteration != ITER_DONE) {
			trained = false;
			break;
		}
	}

	return true;
}

void ProcNormalize::save()
{
	XMLDocument xml(trainer->trainFileName(this, "xml"), true);
	DOMDocument *doc = xml.createDocument("ProcNormalize");
	XMLDocument::writeAttribute(doc->getDocumentElement(), "version", 2);

	for(std::vector<PDF>::const_iterator iter = pdfs.begin();
	    iter != pdfs.end(); iter++) {
		DOMElement *elem = doc->createElement(XMLUniStr("pdf"));
		xml.getRootNode()->appendChild(elem);

		unsigned int i = iter - pdfs.begin();
		const SourceVariable *var = getInputs().get()[i];
		XMLDocument::writeAttribute(elem, "source",
				(const char*)var->getSource()->getName());
		XMLDocument::writeAttribute(elem, "name",
				(const char*)var->getName());

		XMLDocument::writeAttribute(elem, "lower", iter->range.min);
		XMLDocument::writeAttribute(elem, "upper", iter->range.max);

		for(std::vector<double>::const_iterator iter2 =
		    iter->distr.begin(); iter2 != iter->distr.end(); iter2++) {
			DOMElement *value =
					doc->createElement(XMLUniStr("value"));
			elem->appendChild(value);	

			XMLDocument::writeContent<double>(value, doc, *iter2);
		}
	}
}

} // anonymous namespace

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

	virtual void configure(DOMElement *elem) override;
	virtual Calibration::VarProcessor *getCalibration() const override;

	virtual void trainBegin() override;
	virtual void trainData(const std::vector<double> *values,
	                       bool target, double weight) override;
	virtual void trainEnd() override;

	virtual bool load() override;
	virtual void save() override;

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
	int			categoryIdx;
	unsigned int		nCategories;
};

static ProcNormalize::Registry registry("ProcNormalize");

ProcNormalize::ProcNormalize(const char *name, const AtomicId *id,
                             MVATrainer *trainer) :
	TrainProcessor(name, id, trainer),
	categoryIdx(-1),
	nCategories(1)
{
}

ProcNormalize::~ProcNormalize()
{
}

void ProcNormalize::configure(DOMElement *elem)
{
	int i = 0;
	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		DOMElement *elem = static_cast<DOMElement*>(node);

		XMLSimpleStr nodeName(node->getNodeName());

		if (std::strcmp(nodeName, "category") != 0) {
			i++;
			continue;
		}

		if (categoryIdx >= 0)
			throw cms::Exception("ProcNormalize")
				<< "More than one category variable given."
				<< std::endl;


		unsigned int count = XMLDocument::readAttribute<unsigned int>(
								elem, "count");

		categoryIdx = i;
		nCategories = count;
	}

	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		XMLSimpleStr nodeName(node->getNodeName());
		if (std::strcmp(nodeName, "category") == 0)
			continue;

		if (std::strcmp(nodeName, "pdf") != 0)
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

		for(unsigned int i = 0; i < nCategories; i++)
			pdfs.push_back(pdf);
	}

	unsigned int nInputs = getInputs().size();
	if (categoryIdx >= 0)
		nInputs--;

	if (pdfs.size() != nInputs * nCategories)
		throw cms::Exception("ProcNormalize")
			<< "Got " << pdfs.size()
			<< " pdf configs in total for " << nCategories
			<< " categories and " << nInputs
			<< " input varibles (" << (nInputs * nCategories) << " in total)." << std::endl;
}

Calibration::VarProcessor *ProcNormalize::getCalibration() const
{
	Calibration::ProcNormalize *calib = new Calibration::ProcNormalize;

	std::vector<unsigned int> pdfMap;
	for(unsigned int i = 0; i < nCategories; i++)
		for(unsigned int j = i; j < pdfs.size(); j += nCategories)
			pdfMap.push_back(j);

	for(unsigned int i = 0; i < pdfs.size(); i++)
		calib->distr.push_back(pdfs[pdfMap[i]]);

	calib->categoryIdx = categoryIdx;

	return calib;
}

void ProcNormalize::trainBegin()
{
}

void ProcNormalize::trainData(const std::vector<double> *values,
                              bool target, double weight)
{
	int category = 0;
	if (categoryIdx >= 0)
		category = (int)values[categoryIdx].front();
	if (category < 0 || category >= (int)nCategories)
		return;

	int i = 0;
        for(std::vector<PDF>::iterator iter = pdfs.begin() + category;
            iter < pdfs.end(); iter += nCategories, values++) {
		if (i++ == categoryIdx)
			values++;

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
		if (categoryIdx >= 0)
			inputs.erase(inputs.begin() + categoryIdx);

		for(std::vector<PDF>::iterator iter = pdfs.begin();
		    iter != pdfs.end(); iter++) {
			unsigned int idx = iter - pdfs.begin();
			unsigned int catIdx = idx % nCategories;
			unsigned int varIdx = idx / nCategories;
			SourceVariable *var = inputs[varIdx];
			std::string name =
				(const char*)var->getSource()->getName()
				+ std::string("_")
				+ (const char*)var->getName();
			std::string title = name;
			if (categoryIdx >= 0) {
				name += Form("_CAT%d", catIdx);
				title += Form(" (cat. %d)", catIdx);
			}

			unsigned int n = iter->distr.size() - 1;
			double min = iter->range.min -
			             0.5 * iter->range.width() / n;
			double max = iter->range.max +
			             0.5 * iter->range.width() / n;
			TH1F *histo = monitoring->book<TH1F>(name + "_pdf",
				name.c_str(), title.c_str(), n + 1, min, max);
			for(unsigned int i = 0; i < n; i++)
				histo->SetBinContent(i + 1, iter->distr[i]);
		}
	}
}

namespace {
	struct Id {
		AtomicId	source;
		AtomicId	name;
		unsigned int	category;

		inline Id(AtomicId source, AtomicId name,
		          unsigned int category) :
			source(source), name(name), category(category) {}

		inline bool operator == (const Id &other) const
		{
			return source == other.source &&
			       name == other.name &&
			       category == other.category;
		}

		inline bool operator < (const Id &other) const
		{
			if (source < other.source)
				return true;
			if (!(source == other.source))
				return false;
			if (name < other.name)
				return true;
			if (!(name == other.name))
				return false;
			return category < other.category;
		}
	};
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

	std::map<Id, PDF*> pdfMap;

	for(std::vector<PDF>::iterator iter = pdfs.begin();
	    iter != pdfs.end(); ++iter) {
		PDF *ptr = &*iter;
		unsigned int idx = iter - pdfs.begin();
		unsigned int catIdx = idx % nCategories;
		unsigned int varIdx = idx / nCategories;
		if (categoryIdx >= 0 && (int)varIdx >= categoryIdx)
			varIdx++;
		const SourceVariable *var = getInputs().get()[varIdx];
		Id id(var->getSource()->getName(), var->getName(), catIdx);

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
			      				elem, "name"),
			      XMLDocument::readAttribute<unsigned int>(
                                                        elem, "category", 0));
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

		unsigned int idx = iter - pdfs.begin();
		unsigned int catIdx = idx % nCategories;
		unsigned int varIdx = idx / nCategories;
		if (categoryIdx >= 0 && (int)varIdx >= categoryIdx)
			varIdx++;
		const SourceVariable *var = getInputs().get()[varIdx];
		XMLDocument::writeAttribute(elem, "source",
				(const char*)var->getSource()->getName());
		XMLDocument::writeAttribute(elem, "name",
				(const char*)var->getName());
		if (categoryIdx >= 0)
			XMLDocument::writeAttribute(elem, "category", catIdx);

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

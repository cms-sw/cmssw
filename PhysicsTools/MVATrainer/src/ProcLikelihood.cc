#include <algorithm>
#include <iostream>
#include <numeric>
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

class ProcLikelihood : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcLikelihood>::Type Registry;

	ProcLikelihood(const char *name, const AtomicId *id,
	               MVATrainer *trainer);
	~ProcLikelihood() override;

	void configure(DOMElement *elem) override;
	Calibration::VarProcessor *getCalibration() const override;

	void trainBegin() override;
	void trainData(const std::vector<double> *values,
	                       bool target, double weight) override;
	void trainEnd() override;

	bool load() override;
	void save() override;

	struct PDF {
		std::vector<double>		distr;
		Calibration::HistogramD::Range	range;
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

	std::vector<SigBkg>	pdfs;
	std::vector<double>	sigSum;
	std::vector<double>	bkgSum;
	std::vector<double>	bias;
	int			categoryIdx;
	bool			logOutput;
	bool			individual;
	bool			neverUndefined;
	bool			keepEmpty;
	unsigned int		nCategories;
	bool			doCategoryBias;
	bool			doGivenBias;
	bool			doGlobalBias;
	Iteration		iteration;
};

ProcLikelihood::Registry registry("ProcLikelihood");

ProcLikelihood::ProcLikelihood(const char *name, const AtomicId *id,
                               MVATrainer *trainer) :
	TrainProcessor(name, id, trainer),
	categoryIdx(-1),
	logOutput(false),
	individual(false),
	neverUndefined(true),
	keepEmpty(false),
	nCategories(1),
	doCategoryBias(false),
	doGivenBias(false),
	doGlobalBias(false),
	iteration(ITER_FILL)
{
}

ProcLikelihood::~ProcLikelihood()
{
}

void ProcLikelihood::configure(DOMElement *elem)
{
	int i = 0;
	bool first = true;
	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		DOMElement *elem = static_cast<DOMElement*>(node);

		XMLSimpleStr nodeName(node->getNodeName());

		if (std::strcmp(nodeName, "general") == 0) {
			if (!first)
				throw cms::Exception("ProcLikelihood")
					<< "Config tag general needs to come "
					   "first." << std::endl;

			if (XMLDocument::hasAttribute(elem, "bias")) {
				double globalBias =
					XMLDocument::readAttribute<double>(
								elem, "bias");
				bias.push_back(globalBias);
				doGivenBias = true;
			} else
				doGivenBias = false;

			doCategoryBias = XMLDocument::readAttribute<bool>(
						elem, "category_bias", false);
			doGlobalBias = XMLDocument::readAttribute<bool>(
						elem, "global_bias", false);
			logOutput = XMLDocument::readAttribute<bool>(
						elem, "log", false);
			individual = XMLDocument::readAttribute<bool>(
						elem, "individual", false);
			neverUndefined = !XMLDocument::readAttribute<bool>(
						elem, "strict", false);
			keepEmpty = !XMLDocument::readAttribute<bool>(
						elem, "ignore_empty", true);

			first = false;
			continue;
		}
		first = false;

		if (std::strcmp(nodeName, "bias_table") == 0) {
			if (!bias.empty())
				throw cms::Exception("ProcLikelihood")
					<< "Bias can be only specified once."
					<< std::endl;

			for(DOMNode *subNode = node->getFirstChild();
			    subNode; subNode = subNode->getNextSibling()) {
				if (subNode->getNodeType() !=
				    DOMNode::ELEMENT_NODE)
					continue;

				if (std::strcmp(XMLSimpleStr(
						subNode->getNodeName()),
				                "bias") != 0)
					throw cms::Exception("ProcLikelihood")
						<< "Expected bias tag in "
						   "config." << std::endl;

				bias.push_back(
					XMLDocument::readContent<double>(
								subNode));
			}

			continue;
		}

		if (std::strcmp(nodeName, "category") != 0) {
			i++;
			continue;
		}

		if (categoryIdx >= 0)
			throw cms::Exception("ProcLikelihood")
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
		if (std::strcmp(nodeName, "general") == 0 ||
		    std::strcmp(nodeName, "bias_table") == 0 ||
		    std::strcmp(nodeName, "category") == 0)
			continue;

		if (std::strcmp(nodeName, "sigbkg") != 0)
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

		if (XMLDocument::hasAttribute(elem, "lower") &&
		    XMLDocument::hasAttribute(elem, "upper")) {
			pdf.signal.range.min =
				XMLDocument::readAttribute<double>(
								elem, "lower");
			pdf.signal.range.max =
				XMLDocument::readAttribute<double>(
								elem, "upper");
			pdf.background.range = pdf.signal.range;
			pdf.iteration = ITER_FILL;
		} else
			pdf.iteration = ITER_EMPTY;

		for(unsigned int i = 0; i < nCategories; i++)
			pdfs.push_back(pdf);
	}

	unsigned int nInputs = getInputs().size();
	if (categoryIdx >= 0)
		nInputs--;

	sigSum.resize(nCategories);
	bkgSum.resize(nCategories);

	if (!doGivenBias && !bias.empty()) {
		doGivenBias = true;
		if (bias.size() != nCategories)
			throw cms::Exception("ProcLikelihood")
				<< "Invalid number of category bias entries."
				<< std::endl;
	}
	while (doGivenBias && bias.size() < nCategories)
		bias.push_back(bias.front());

	if (pdfs.size() != nInputs * nCategories)
		throw cms::Exception("ProcLikelihood")
			<< "Got " << (pdfs.size() / nCategories)
		        << " pdf configs for " << nInputs
		        << " input variables." << std::endl;
}

Calibration::VarProcessor *ProcLikelihood::getCalibration() const
{
	typedef Calibration::ProcLikelihood Calib;

	Calibration::ProcLikelihood *calib = new Calibration::ProcLikelihood;

	std::vector<unsigned int> pdfMap;
	for(unsigned int i = 0; i < nCategories; i++)
		for(unsigned int j = i; j < pdfs.size(); j += nCategories)
			pdfMap.push_back(j);

	double totalSig = std::accumulate(sigSum.begin(), sigSum.end(), 0.0);
	double totalBkg = std::accumulate(bkgSum.begin(), bkgSum.end(), 0.0);

	for(unsigned int i = 0; i < pdfs.size(); i++) {
		const SigBkg *iter = &pdfs[pdfMap[i]];
		Calibration::ProcLikelihood::SigBkg pdf;

		pdf.signal = Calibration::HistogramF(iter->signal.distr.size(),
		                                     iter->signal.range);
		double factor = std::accumulate(iter->signal.distr.begin(),
			                        iter->signal.distr.end(), 0.0);
		if (factor < 1e-20)
			factor = 1.0;
		else
			factor = 1.0 / factor;
		std::vector<double> values(iter->signal.distr.size() + 2);
		std::transform(iter->signal.distr.begin(),
		               iter->signal.distr.end(),
		               values.begin() + 1,
		               [&](auto const& c) {return c * factor;});
		pdf.signal.setValues(values);

		pdf.background =
			Calibration::HistogramF(iter->background.distr.size(),
			                        iter->background.range.min,
			                        iter->background.range.max);
		factor = std::accumulate(iter->background.distr.begin(),
		                         iter->background.distr.end(), 0.0);
		if (factor < 1e-20)
			factor = 1.0;
		else
			factor = 1.0 / factor;
		std::transform(iter->background.distr.begin(),
		               iter->background.distr.end(),
		               values.begin() + 1,
		               [&](auto const& c){return c * factor;});
		pdf.background.setValues(values);

		pdf.useSplines = true;

		calib->pdfs.push_back(pdf);
	}

	calib->categoryIdx = categoryIdx;
	calib->logOutput = logOutput;
	calib->individual = individual;
	calib->neverUndefined = neverUndefined;
	calib->keepEmpty = keepEmpty;

	if (doGlobalBias || doCategoryBias || doGivenBias) {
		for(unsigned int i = 0; i < nCategories; i++) {
			double bias = doGlobalBias
						? totalSig / totalBkg
						: 1.0;
			if (doGivenBias)
				bias *= this->bias[i];
			if (doCategoryBias)
				bias *= (sigSum[i] / totalSig) /
				        (bkgSum[i] / totalBkg);
			calib->bias.push_back(bias);
		}
	}

	return calib;
}

void ProcLikelihood::trainBegin()
{
}

void ProcLikelihood::trainData(const std::vector<double> *values,
                               bool target, double weight)
{
	int category = 0;
	if (categoryIdx >= 0)
		category = (int)values[categoryIdx].front();
	if (category < 0 || category >= (int)nCategories)
		return;

	if (iteration == ITER_FILL) {
		if (target)
			sigSum[category] += weight;
		else
			bkgSum[category] += weight;
	}

	int i = 0;
	for(std::vector<SigBkg>::iterator iter = pdfs.begin() + category;
	    iter < pdfs.end(); iter += nCategories, values++) {
		if (i++ == categoryIdx)
			values++;

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
					         *value);
				iter->signal.range.max =
					std::max(iter->signal.range.max,
					         *value);
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

void smoothArray(unsigned int n, double *values, unsigned int nTimes)
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
	if (iteration == ITER_FILL)
		iteration = ITER_DONE;

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

	if (done && monitoring) {
		std::vector<SourceVariable*> inputs = getInputs().get();
		if (categoryIdx >= 0)
			inputs.erase(inputs.begin() + categoryIdx);

		for(std::vector<SigBkg>::iterator iter = pdfs.begin();
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

			unsigned int n = iter->signal.distr.size() - 1;
			double min = iter->signal.range.min -
			             0.5 * iter->signal.range.width() / n;
			double max = iter->signal.range.max +
			             0.5 * iter->signal.range.width() / n;
			TH1F *histo = monitoring->book<TH1F>(name + "_sig",
				(name + "_sig").c_str(),
				(title + " signal").c_str(), n + 1, min, max);
			for(unsigned int i = 0; i < n; i++)
				histo->SetBinContent(
					i + 1, iter->signal.distr[i]);

			n = iter->background.distr.size() - 1;
			min = iter->background.range.min -
			      0.5 * iter->background.range.width() / n;
			max = iter->background.range.max +
			      0.5 * iter->background.range.width() / n;
			histo = monitoring->book<TH1F>(name + "_bkg",
				(name + "_bkg").c_str(),
				(title + " background").c_str(),
				n + 1, min, max);
			for(unsigned int i = 0; i < n; i++)
				histo->SetBinContent(
					i + 1, iter->background.distr[i]);
		}
	}
}

void xmlParsePDF(ProcLikelihood::PDF &pdf, DOMElement *elem)
{
	if (!elem ||
	    std::strcmp(XMLSimpleStr(elem->getNodeName()), "pdf") != 0)
		throw cms::Exception("ProcLikelihood")
			<< "Expected pdf tag in sigbkg train data."
			<< std::endl;

	pdf.range.min = XMLDocument::readAttribute<double>(elem, "lower");
	pdf.range.max = XMLDocument::readAttribute<double>(elem, "upper");

	pdf.distr.clear();
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

bool ProcLikelihood::load()
{
	std::string filename = trainer->trainFileName(this, "xml");
	if (!exists(filename))
		return false;

	XMLDocument xml(filename);
	DOMElement *elem = xml.getRootNode();
	if (std::strcmp(XMLSimpleStr(elem->getNodeName()),
	                             "ProcLikelihood") != 0)
		throw cms::Exception("ProcLikelihood")
			<< "XML training data file has bad root node."
			<< std::endl;

	unsigned int version = XMLDocument::readAttribute<unsigned int>(
							elem, "version", 1);

	if (version < 1 || version > 2)
		throw cms::Exception("ProcLikelihood")
			<< "Unsupported version " << version
			<< "in train file." << std::endl;

	DOMNode *node;
	for(node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                "categories") != 0)
			throw cms::Exception("ProcLikelihood")
				<< "Expected categories tag in train file."
				<< std::endl;

		unsigned int i = 0;
		for(DOMNode *subNode = node->getFirstChild();
		    subNode; subNode = subNode->getNextSibling()) {
			if (subNode->getNodeType() != DOMNode::ELEMENT_NODE)
				continue;

			if (i >= nCategories)
				throw cms::Exception("ProcLikelihood")
					<< "Too many categories in train "
				           "file." << std::endl;

			if (std::strcmp(XMLSimpleStr(subNode->getNodeName()),
			                "category") != 0)
				throw cms::Exception("ProcLikelihood")
					<< "Expected category tag in train "
				           "file." << std::endl;

			elem = static_cast<DOMElement*>(subNode);

			sigSum[i] = XMLDocument::readAttribute<double>(
							elem, "signal");
			bkgSum[i] = XMLDocument::readAttribute<double>(
							elem, "background");
			i++;
		}
		if (i < nCategories)
			throw cms::Exception("ProcLikelihood")
				<< "Too few categories in train file."
				<< std::endl;

		break;
	}

	std::map<Id, SigBkg*> pdfMap;

	for(std::vector<SigBkg>::iterator iter = pdfs.begin();
	    iter != pdfs.end(); ++iter) {
		SigBkg *ptr = &*iter;
		unsigned int idx = iter - pdfs.begin();
		unsigned int catIdx = idx % nCategories;
		unsigned int varIdx = idx / nCategories;
		if (categoryIdx >= 0 && (int)varIdx >= categoryIdx)
			varIdx++;
		const SourceVariable *var = getInputs().get()[varIdx];
		Id id(var->getSource()->getName(), var->getName(), catIdx);

		pdfMap[id] = ptr;
	}

	std::vector<SigBkg>::iterator cur = pdfs.begin();

	for(node = node->getNextSibling();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                "sigbkg") != 0)
			throw cms::Exception("ProcLikelihood")
				<< "Expected sigbkg tag in train file."
				<< std::endl;
		elem = static_cast<DOMElement*>(node);

		SigBkg *pdf = nullptr;
		switch(version) {
		    case 1:
			if (cur == pdfs.end())
				throw cms::Exception("ProcLikelihood")
					<< "Superfluous SigBkg in train data."
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
			std::map<Id, SigBkg*>::const_iterator pos =
							pdfMap.find(id);
			if (pos == pdfMap.end())
				continue;
			else
				pdf = pos->second;
		    }	break;
		}

		for(node = elem->getFirstChild();
		    node && node->getNodeType() != DOMNode::ELEMENT_NODE;
		    node = node->getNextSibling());
		DOMElement *elemSig =
				node ? static_cast<DOMElement*>(node) : nullptr;

		for(node = node->getNextSibling();
		    node && node->getNodeType() != DOMNode::ELEMENT_NODE;
		    node = node->getNextSibling());
		while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
			node = node->getNextSibling();
		DOMElement *elemBkg =
				node ? static_cast<DOMElement*>(node) : nullptr;

		for(node = node->getNextSibling();
		    node && node->getNodeType() != DOMNode::ELEMENT_NODE;
		    node = node->getNextSibling());
		if (node)
			throw cms::Exception("ProcLikelihood")
				<< "Superfluous tags in sigbkg train data."
				<< std::endl;

		xmlParsePDF(pdf->signal, elemSig);
		xmlParsePDF(pdf->background, elemBkg);

		pdf->iteration = ITER_DONE;

		node = elem;
	}

	if (version == 1 && cur != pdfs.end())
		throw cms::Exception("ProcLikelihood")
			<< "Missing SigBkg in train data." << std::endl;

	iteration = ITER_DONE;
	trained = true;
	for(std::vector<SigBkg>::const_iterator iter = pdfs.begin();
	    iter != pdfs.end(); ++iter) {
		if (iter->iteration != ITER_DONE) {
			trained = false;
			break;
		}
	}

	return true;
}

DOMElement *xmlStorePDF(DOMDocument *doc,
                               const ProcLikelihood::PDF &pdf)
{
	DOMElement *elem = doc->createElement(XMLUniStr("pdf"));

	XMLDocument::writeAttribute(elem, "lower", pdf.range.min);
	XMLDocument::writeAttribute(elem, "upper", pdf.range.max);

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
	XMLDocument::writeAttribute(doc->getDocumentElement(), "version", 2);

	DOMElement *elem = doc->createElement(XMLUniStr("categories"));
	xml.getRootNode()->appendChild(elem);
	for(unsigned int i = 0; i < nCategories; i++) {
		DOMElement *category = doc->createElement(XMLUniStr("category"));
		elem->appendChild(category);
		XMLDocument::writeAttribute(category, "signal", sigSum[i]);
		XMLDocument::writeAttribute(category, "background", bkgSum[i]);
	}

	for(std::vector<SigBkg>::const_iterator iter = pdfs.begin();
	    iter != pdfs.end(); iter++) {
		elem = doc->createElement(XMLUniStr("sigbkg"));
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

		elem->appendChild(xmlStorePDF(doc, iter->signal));
		elem->appendChild(xmlStorePDF(doc, iter->background));
	}
}

} // anonymous namespace

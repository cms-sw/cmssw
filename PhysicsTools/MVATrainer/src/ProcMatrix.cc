#include <iostream>
#include <cstring>
#include <vector>
#include <memory>

#include <xercesc/dom/DOM.hpp>

#include <TMatrixD.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"
#include "PhysicsTools/MVATrainer/interface/LeastSquares.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcMatrix : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcMatrix>::Type Registry;

	ProcMatrix(const char *name, const AtomicId *id,
	           MVATrainer *trainer);
	virtual ~ProcMatrix();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

	virtual void trainBegin();
	virtual void trainData(const std::vector<double> *values,
	                       bool target, double weight);
	virtual void trainEnd();

	virtual bool load();
	virtual void save();

    protected:
	virtual void *requestObject(const std::string &name) const;

    private:
	enum Iteration {
		ITER_FILL,
		ITER_DONE
	} iteration;

	std::auto_ptr<LeastSquares>	ls;
	std::vector<double>		vars;
	bool				fillSignal;
	bool				fillBackground;
};

static ProcMatrix::Registry registry("ProcMatrix");

ProcMatrix::ProcMatrix(const char *name, const AtomicId *id,
                             MVATrainer *trainer) :
	TrainProcessor(name, id, trainer),
	iteration(ITER_FILL), fillSignal(true), fillBackground(true)
{
}

ProcMatrix::~ProcMatrix()
{
}

void ProcMatrix::configure(DOMElement *elem)
{
	ls = std::auto_ptr<LeastSquares>(new LeastSquares(getInputs().size()));

	DOMNode *node = elem->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node)
		return;

	if (std::strcmp(XMLSimpleStr(node->getNodeName()), "fill") != 0)
		throw cms::Exception("ProcMatrix")
				<< "Expected fill tag in config section."
				<< std::endl;

	elem = static_cast<DOMElement*>(node);

	fillSignal =
		XMLDocument::readAttribute<bool>(elem, "signal", false);
	fillBackground =
		XMLDocument::readAttribute<bool>(elem, "background", false);

	node = node->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcMatrix")
			<< "Superfluous tags in config section."
			<< std::endl;

	if (!fillSignal && !fillBackground)
		throw cms::Exception("ProcMatrix")
			<< "Filling neither background nor signal in config."
			<< std::endl;
}

Calibration::VarProcessor *ProcMatrix::getCalibration() const
{
	Calibration::ProcMatrix *calib = new Calibration::ProcMatrix;

	unsigned int n = ls->getSize();
	const TMatrixD &rotation = ls->getRotation();

	calib->matrix.rows = n;
	calib->matrix.columns = n;

	for(unsigned int i = 0; i < n; i++)
		for(unsigned int j = 0; j < n; j++)
			calib->matrix.elements.push_back(rotation(i, j));

	return calib;
}

void ProcMatrix::trainBegin()
{
	if (iteration == ITER_FILL)
		vars.resize(ls->getSize());
}

void ProcMatrix::trainData(const std::vector<double> *values,
                           bool target, double weight)
{
	if (iteration != ITER_FILL)
		return;

	if (!(target ? fillSignal : fillBackground))
		return;

	for(unsigned int i = 0; i < ls->getSize(); i++, values++)
		vars[i] = values->front();

	ls->add(vars, target, weight);
}

void ProcMatrix::trainEnd()
{
	switch(iteration) {
	    case ITER_FILL:
		vars.clear();
		ls->calculate();

		iteration = ITER_DONE;
		trained = true;
		break;
	    default:
		/* shut up */;
	}
}

void *ProcMatrix::requestObject(const std::string &name) const
{
	if (name == "linearAnalyzer")
		return static_cast<void*>(ls.get());

	return 0;
}

bool ProcMatrix::load()
{
	std::string filename = trainer->trainFileName(this, "xml");
	if (!exists(filename))
		return false;

	XMLDocument xml(filename);
	DOMElement *elem = xml.getRootNode();
	if (std::strcmp(XMLSimpleStr(elem->getNodeName()), "ProcMatrix") != 0)
		throw cms::Exception("ProcMatrix")
			<< "XML training data file has bad root node."
			<< std::endl;

	DOMNode *node = elem->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node)
		throw cms::Exception("ProcMatrix")
			<< "Train data file empty." << std::endl;

	ls->load(static_cast<DOMElement*>(node));

	node = elem->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcMatrix")
			<< "Train data file contains superfluous tags."
			<< std::endl;

	iteration = ITER_DONE;
	trained = true;
	return true;
}

void ProcMatrix::save()
{
	XMLDocument xml(trainer->trainFileName(this, "xml"), true);
	DOMDocument *doc = xml.createDocument("ProcMatrix");

	xml.getRootNode()->appendChild(ls->save(doc));
}

} // anonymous namespace

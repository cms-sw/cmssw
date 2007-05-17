#include <iostream>
#include <vector>
#include <memory>

#include <xercesc/dom/DOM.hpp>

#include <TMatrixD.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/Processor.h"
#include "PhysicsTools/MVATrainer/interface/LeastSquares.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcMatrix : public Processor {
    public:
	typedef Processor::Registry<ProcMatrix>::Type Registry;

	ProcMatrix(const char *name, const AtomicId *id,
	           MVATrainer *trainer);
	virtual ~ProcMatrix();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalib() const;

	virtual void trainBegin();
	virtual void trainData(const std::vector<double> *values, bool target);
	virtual void trainEnd();

    protected:
	virtual void *requestObject(const std::string &name) const;

    private:
	enum Iteration {
		ITER_FILL,
		ITER_DONE
	} iteration;

	bool load();
	void save() const;

	std::auto_ptr<LeastSquares>		ls;
	std::auto_ptr< std::vector<double> >	vars;
};

static ProcMatrix::Registry registry("ProcMatrix");

ProcMatrix::ProcMatrix(const char *name, const AtomicId *id,
                             MVATrainer *trainer) :
	Processor(name, id, trainer),
	iteration(ITER_FILL)
{
}

ProcMatrix::~ProcMatrix()
{
}

void ProcMatrix::configure(DOMElement *elem)
{
	ls = std::auto_ptr<LeastSquares>(new LeastSquares(getInputs().size()));

	if (load()) {
		iteration = ITER_DONE;
		trained = true;
		std::cout << "ProcNormalize configuration for \""
		          << getName() << "\" loaded from file."
		          << std::endl;
	}
}

Calibration::VarProcessor *ProcMatrix::getCalib() const
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
		vars = std::auto_ptr< std::vector<double> >(
				new std::vector<double>(ls->getSize()));
}

void ProcMatrix::trainData(const std::vector<double> *values, bool target)
{
	if (iteration != ITER_FILL)
		return;

	for(unsigned int i = 0; i < ls->getSize(); i++, values++)
		vars->at(i) = values->front();

	ls->add(*vars, target);
}

void ProcMatrix::trainEnd()
{
	switch(iteration) {
	    case ITER_FILL:
		vars.reset();
		ls->calculate();

		save();
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
	std::auto_ptr<XMLDocument> xml;

	try {
		xml = std::auto_ptr<XMLDocument>(new XMLDocument(
				trainer->trainFileName(this, "xml")));
	} catch(...) {
		return false;
	}

	DOMElement *elem = xml->getRootNode();
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

	return true;
}

void ProcMatrix::save() const
{
	XMLDocument xml(trainer->trainFileName(this, "xml"), true);
	DOMDocument *doc = xml.createDocument("ProcMatrix");

	xml.getRootNode()->appendChild(ls->save(doc));
}

} // anonymous namespace

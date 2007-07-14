#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <memory>

#include <xercesc/dom/DOM.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcMLP : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcMLP>::Type Registry;

	ProcMLP(const char *name, const AtomicId *id,
	        MVATrainer *trainer);
	virtual ~ProcMLP();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

	virtual void trainBegin();
	virtual void trainData(const std::vector<double> *values,
	                       bool target, double weight);
	virtual void trainEnd();

	virtual bool load();
	virtual void cleanup();

    private:
	enum Iteration {
		ITER_TRAIN,
		ITER_DONE
	} iteration;

	std::string		layout;
	unsigned int		steps;
	unsigned int		count, row;
	std::vector<double>	vars;
	std::vector<double>	targets;
	bool			needCleanup;
};

static ProcMLP::Registry registry("ProcMLP");

ProcMLP::ProcMLP(const char *name, const AtomicId *id,
                 MVATrainer *trainer) :
	TrainProcessor(name, id, trainer),
	iteration(ITER_TRAIN),
	count(0),
	needCleanup(false)
{
}

ProcMLP::~ProcMLP()
{
}

void ProcMLP::configure(DOMElement *elem)
{
	std::vector<SourceVariable*> inputs = getInputs().get();

	DOMNode *node = elem->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node)
		throw cms::Exception("ProcMLP")
			<< "Expected MLP config in config section."
			<< std::endl;

	if (std::strcmp(XMLSimpleStr(node->getNodeName()), "config") != 0)
		throw cms::Exception("ProcMLP")
				<< "Expected config tag in config section."
				<< std::endl;

	elem = static_cast<DOMElement*>(node);

	steps = XMLDocument::readAttribute<unsigned int>(elem, "steps");

	layout = (const char*)XMLSimpleStr(node->getTextContent());

	node = node->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcMLP")
			<< "Superfluous tags in config section."
			<< std::endl;

	vars.resize(getInputs().size());
	targets.resize(getOutputs().size());
}

bool ProcMLP::load()
{
	bool ok = false;
	/* test for weights file */ {
		std::ifstream in(trainer->trainFileName(this, "txt").c_str());
		ok = in.good();
	}

	if (!ok)
		return false;

	iteration = ITER_DONE;
	trained = true;
	return true;
}

Calibration::VarProcessor *ProcMLP::getCalibration() const
{
	Calibration::ProcMLP *calib = new Calibration::ProcMLP;

	std::string fileName = trainer->trainFileName(this, "txt");
	std::ifstream in(fileName.c_str(), std::ios::binary | std::ios::in);
	if (!in.good())
		throw cms::Exception("ProcMLP")
			<< "Weights file " << fileName
			<< "cannot be opened for reading." << std::endl;

	char linebuf[128];
	linebuf[127] = 0;
	in.getline(linebuf, 127);
	if (std::strncmp(linebuf, "# network structure ", 20) != 0)
		throw cms::Exception("ProcMLP")
			<< "Weights file " << fileName
			<< "has invalid header." << std::endl;

	std::istringstream is(linebuf + 20);
	std::vector<unsigned int> layout;
	for(;;) {
		unsigned int layer = 0;
		is >> layer;
		if (!layer)
			break;
		layout.push_back(layer);
	}

	if (layout.size() < 2 || layout.front() != getInputs().size()
	    || layout.back() != 1)
		throw cms::Exception("ProcMLP")
			<< "Weights file " << fileName
			<< "network layout does not match." << std::endl;

	in.getline(linebuf, 127);

	for(unsigned int layer = 1; layer < layout.size(); layer++) {
		Calibration::ProcMLP::Layer layerConf;

		for(unsigned int i = 0; i < layout[layer]; i++) {
			Calibration::ProcMLP::Neuron neuron;

			for(unsigned int j = 0; j <= layout[layer - 1]; j++) {
				in.getline(linebuf, 127);
				std::istringstream ss(linebuf);
				double weight;
				ss >> weight;

				if (j == 0)
					neuron.first = weight;
				else
					neuron.second.push_back(weight);
			}
			layerConf.first.push_back(neuron);
		}
		layerConf.second = layer < layout.size() - 1;

		calib->layers.push_back(layerConf);
	}

	in.close();

	return calib;
}

void ProcMLP::trainBegin()
{
	switch(iteration) {
	    case ITER_TRAIN:
		throw cms::Exception("ProcMLP")
			<< "Actual training for ProcMLP not provided"
			   "inside CMSSW. Please provide network weights"
			   "file in mlpfit format." << std::endl;
		break;
	    default:
		/* shut up */;
	}
}

void ProcMLP::trainData(const std::vector<double> *values,
                        bool target, double weight)
{
}

void ProcMLP::trainEnd()
{
	switch(iteration) {
	    default:
		/* shut up */;
	}
}

void ProcMLP::cleanup()
{
	if (!needCleanup)
		return;

	std::remove(trainer->trainFileName(this, "txt").c_str());
}

} // anonymous namespace

#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <memory>
#include <cmath>

#include <TRandom.h>

#include <xercesc/dom/DOM.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"

#include "MLP.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcMLP : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcMLP>::Type Registry;

	ProcMLP(const char *name, const AtomicId *id,
	        MVATrainer *trainer);
	~ProcMLP() override;

	void configure(DOMElement *elem) override;
	Calibration::VarProcessor *getCalibration() const override;

	void trainBegin() override;
	void trainData(const std::vector<double> *values,
	                       bool target, double weight) override;
	void trainEnd() override;

	bool load() override;
	void cleanup() override;

    private:
	void runMLPTrainer();

	enum Iteration {
		ITER_COUNT,
		ITER_TRAIN,
		ITER_WAIT,
		ITER_DONE
	} iteration;

	std::string		layout;
	unsigned int		steps;
	unsigned int		count, row;
	double			weightSum;
	std::unique_ptr<MLP>	mlp;
	std::vector<double>	vars;
	std::vector<double>	targets;
	bool			needCleanup;
	int			boost;
	TRandom			rand;
	double			limiter;
};

ProcMLP::Registry registry("ProcMLP");

ProcMLP::ProcMLP(const char *name, const AtomicId *id,
                 MVATrainer *trainer) :
	TrainProcessor(name, id, trainer),
	iteration(ITER_COUNT),
	count(0),
	weightSum(0.0),
	needCleanup(false),
	boost(-1),
	limiter(0.0)
{
}

ProcMLP::~ProcMLP()
{
}

void ProcMLP::configure(DOMElement *elem)
{
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

	boost = XMLDocument::readAttribute<int>(elem, "boost", -1);
	limiter = XMLDocument::readAttribute<double>(elem, "limiter", 0);
	steps = XMLDocument::readAttribute<unsigned int>(elem, "steps");

	layout = (const char*)XMLSimpleStr(node->getTextContent());

	node = node->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcMLP")
			<< "Superfluous tags in config section."
			<< std::endl;

	vars.resize(getInputs().size() - (boost >= 0 ? 1 : 0));
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

//	mlp->load(trainer->trainFileName(this, "txt"));
	iteration = ITER_DONE;
	trained = true;
	return true;
}

Calibration::VarProcessor *ProcMLP::getCalibration() const
{
	Calibration::ProcMLP *calib = new Calibration::ProcMLP;

	std::unique_ptr<MLP> mlp(new MLP(getInputs().size() - (boost >= 0 ? 1 : 0),
	                               getOutputs().size(), layout));
	mlp->load(trainer->trainFileName(this, "txt"));

	std::string fileName = trainer->trainFileName(this, "txt");
	std::ifstream in(fileName.c_str(), std::ios::binary | std::ios::in);
	if (!in.good())
		throw cms::Exception("ProcMLP")
			<< "Weights file " << fileName
			<< "cannot be opened for reading." << std::endl;

	char linebuf[128];
	linebuf[127] = 0;
	do
		in.getline(linebuf, 127);
	while(linebuf[0] == '#');

	int layers = mlp->getLayers();
	const int *neurons = mlp->getLayout();

	for(int layer = 1; layer < layers; layer++) {
		Calibration::ProcMLP::Layer layerConf;

		for(int i = 0; i < neurons[layer]; i++) {
			Calibration::ProcMLP::Neuron neuron;

			for(int j = 0; j <= neurons[layer - 1]; j++) {
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
		layerConf.second = layer < layers - 1;

		calib->layers.push_back(layerConf);
	}

	in.close();

	return calib;
}

void ProcMLP::trainBegin()
{
	rand.SetSeed(65539);
	switch(iteration) {
	    case ITER_COUNT:
		count = 0;
		weightSum = 0.0;
		break;
	    case ITER_TRAIN:
		try {
			mlp = std::unique_ptr<MLP>(
					new MLP(getInputs().size() - (boost >= 0 ? 1 : 0),
					getOutputs().size(), layout));
			mlp->init(count);
			row = 0;
		} catch(cms::Exception const&) {
			// MLP probably busy (or layout invalid, aaaack)
			iteration = ITER_WAIT;
		}
		break;
	    default:
		/* shut up */;
	}
}

void ProcMLP::trainData(const std::vector<double> *values,
                        bool target, double weight)
{
	if (boost >= 0) {
		double x = values[boost][0];
		if (target)
			weight *= 1.0 + 0.02 * std::exp(5.0 * (1.0 - x));
		else
			weight *= 1.0 + 0.1 * std::exp(5.0 * x);
	}

	if (weight < limiter) {
		if (rand.Uniform(limiter) > weight)
			return;
		weight = limiter;
	}

	if (iteration == ITER_COUNT)
		count++;
   weightSum += weight; 

	if (iteration != ITER_TRAIN)
		return;

	for(unsigned int i = 0; i < vars.size(); i++, values++) {
		if ((int)i == boost)
			values++;
		vars[i] = values->front();
	}

	for(unsigned int i = 0; i < targets.size(); i++)
		targets[i] = target;

	mlp->set(row++, &vars.front(), &targets.front(), weight);
}

void ProcMLP::runMLPTrainer()
{
	for(unsigned int i = 0; i < steps; i++) {
		double error = mlp->train();
		if ((i % 10) == 0)
			std::cout << "Training MLP epoch " << mlp->getEpoch()
			          << ", rel chi^2: " << (error / weightSum)
			          << std::endl;
	}
}

void ProcMLP::trainEnd()
{
	switch(iteration) {
	    case ITER_COUNT:
	    case ITER_WAIT:
		iteration = ITER_TRAIN;
		std::cout << "Training with " << count << " events. "
		              "(weighted " << weightSum << ")" << std::endl;
		break;
	    case ITER_TRAIN:
		runMLPTrainer();
		mlp->save(trainer->trainFileName(this, "txt"));
		mlp->clear();
		mlp.reset();
		needCleanup = true;
		iteration = ITER_DONE;
		trained = true;
		break;
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

MVA_TRAINER_DEFINE_PLUGIN(ProcMLP);

} // anonymous namespace

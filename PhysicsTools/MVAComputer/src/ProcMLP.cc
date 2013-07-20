// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcMLP
// 

// Implementation:
//     An evaluator for a feed-forward neural net (multi-layer perceptron).
//     Each layer has (n + 1) x m weights for n input neurons, 1 bias
//     and m neurons. Also each layer can select between linear and logistic
//     activation function. The output from the last layer is returned.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcMLP.cc,v 1.4 2009/06/03 09:50:14 saout Exp $
//

#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cmath>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcMLP : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcMLP,
					Calibration::ProcMLP> Registry;

	ProcMLP(const char *name,
	         const Calibration::ProcMLP *calib,
	         const MVAComputer *computer);
	virtual ~ProcMLP() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;
	virtual std::vector<double> deriv(
				ValueIterator iter, unsigned int n) const;

    private:
	struct Layer {
		Layer(const Calibration::ProcMLP::Layer &calib);
		Layer(const Layer &orig) :
			inputs(orig.inputs), neurons(orig.neurons),
			coeffs(orig.coeffs), sigmoid(orig.sigmoid) {}

		unsigned int		inputs;
		unsigned int		neurons;
		std::vector<double>	coeffs;
		bool			sigmoid;
	};

	std::vector<Layer>	layers;
	unsigned int		maxTmp;
};

static ProcMLP::Registry registry("ProcMLP");

ProcMLP::Layer::Layer(const Calibration::ProcMLP::Layer &calib) :
	inputs(calib.first.front().second.size()),
	neurons(calib.first.size()),
	sigmoid(calib.second)
{
	typedef Calibration::ProcMLP::Neuron Neuron;

	coeffs.resize(neurons * (inputs + 1));
	std::vector<double>::iterator inserter = coeffs.begin();

	for(std::vector<Neuron>::const_iterator iter = calib.first.begin();
	    iter != calib.first.end(); iter++) {
		*inserter++ = iter->first;

		if (iter->second.size() != inputs)
			throw cms::Exception("ProcMLPInput")
				<< "ProcMLP neuron layer inconsistent."
				<< std::endl;

		inserter = std::copy(iter->second.begin(), iter->second.end(),
		                     inserter);
	}
}

ProcMLP::ProcMLP(const char *name,
                   const Calibration::ProcMLP *calib,
                   const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	maxTmp(0)
{
	std::copy(calib->layers.begin(), calib->layers.end(),
	          std::back_inserter(layers));

	for(unsigned int i = 0; i < layers.size(); i++) {
		maxTmp = std::max<unsigned int>(maxTmp, layers[i].neurons);
		if (i > 0 && layers[i - 1].neurons != layers[i].inputs)
			throw cms::Exception("ProcMLPInput")
				<< "ProcMLP neuron layers do not connect "
				"properly." << std::endl;
	}
}

void ProcMLP::configure(ConfIterator iter, unsigned int n)
{
	if (n != layers.front().inputs)
		return;

	for(unsigned int i = 0; i < n; i++)
		iter++(Variable::FLAG_NONE);

	for(unsigned int i = 0; i < layers.back().neurons; i++)
		iter << Variable::FLAG_NONE;
}

void ProcMLP::eval(ValueIterator iter, unsigned int n) const
{
	double *tmp = (double*)alloca(2 * maxTmp * sizeof(double));
	bool flip = false;

	for(double *pos = tmp; iter; iter++, pos++)
		*pos = *iter;

	double *output = 0;
	for(std::vector<Layer>::const_iterator layer = layers.begin();
	    layer != layers.end(); layer++, flip = !flip) {
		const double *input = &tmp[flip ? maxTmp : 0];
		output = &tmp[flip ? 0 : maxTmp];
		std::vector<double>::const_iterator coeff =
							layer->coeffs.begin();
		for(unsigned int i = 0; i < layer->neurons; i++) {
			double sum = *coeff++;
			for(unsigned int j = 0; j < layer->inputs; j++)
				sum += input[j] * *coeff++;
			if (layer->sigmoid)
				sum = 1.0 / (std::exp(-sum) + 1.0);
			*output++ = sum;
		}
	}

	for(const double *pos = &tmp[flip ? maxTmp : 0]; pos < output; pos++)
		iter(*pos);
}

std::vector<double> ProcMLP::deriv(ValueIterator iter, unsigned int n) const
{
	std::vector<double> prevValues, nextValues;
	std::vector<double> prevMatrix, nextMatrix;

	while(iter)
		nextValues.push_back(*iter++);

	unsigned int size = nextValues.size();
	nextMatrix.resize(size * size);
	for(unsigned int i = 0; i < size; i++)
		nextMatrix[i * size + i] = 1.;

	for(std::vector<Layer>::const_iterator layer = layers.begin();
	    layer != layers.end(); layer++) {
		prevValues.clear();
		std::swap(prevValues, nextValues);
		prevMatrix.clear();
		std::swap(prevMatrix, nextMatrix);

		std::vector<double>::const_iterator coeff =
							layer->coeffs.begin();
		for(unsigned int i = 0; i < layer->neurons; i++) {
			double sum = *coeff++;
			for(unsigned int j = 0; j < layer->inputs; j++)
				sum += prevValues[j] * *coeff++;

			double deriv;
			if (layer->sigmoid) {
				double e = std::exp(-sum);
				sum = 1.0 / (e + 1.0);
				deriv = 1.0 / (e + 1.0/e + 2.0);
			} else
				deriv = 1.0;

			nextValues.push_back(sum);

			for(unsigned int k = 0; k < size; k++) {
				sum = 0.0;
				coeff -= layer->inputs;
				for(unsigned int j = 0; j < layer->inputs; j++)
					sum += prevMatrix[j * size + k] *
					       *coeff++;
				nextMatrix.push_back(sum * deriv);
			}
		}
	}

	return nextMatrix;
}

} // anonymous namespace

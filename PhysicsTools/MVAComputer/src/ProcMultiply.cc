// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     
// 

// Implementation:
//     Multiplies n input variables to produce one output variable.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcMultiply.cc,v 1.5 2009/06/03 09:50:14 saout Exp $
//

#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcMultiply : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcMultiply,
					Calibration::ProcMultiply> Registry;

	ProcMultiply(const char *name,
	           const Calibration::ProcMultiply *calib,
	           const MVAComputer *computer);
	virtual ~ProcMultiply() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;
	virtual std::vector<double> deriv(
				ValueIterator iter, unsigned int n) const;

    private:
	typedef std::vector<unsigned int>	Config;

	unsigned int				in;
	std::vector<Config>			out;
};

static ProcMultiply::Registry registry("ProcMultiply");

ProcMultiply::ProcMultiply(const char *name,
                       const Calibration::ProcMultiply *calib,
                       const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	in(calib->in)
{
	std::copy(calib->out.begin(), calib->out.end(),
	          std::back_inserter(out));
}

void ProcMultiply::configure(ConfIterator iter, unsigned int n)
{
	if (in != n)
		return;

	for(unsigned int i = 0; i < in; i++)
		iter++(Variable::FLAG_NONE);

	for(unsigned int i = 0; i < out.size(); i++)
		iter << Variable::FLAG_NONE;
}

void ProcMultiply::eval(ValueIterator iter, unsigned int n) const
{
	double *values = (double*)alloca(in * sizeof(double));
	for(double *pos = values; iter; iter++, pos++) {
		if (iter.size() != 1)
			throw cms::Exception("ProcMultiply")
				<< "Special input variable encountered "
				   "at index " << (pos - values) << "."
				<< std::endl;
		*pos = *iter;
	}

	for(std::vector<Config>::const_iterator config = out.begin();
	    config != out.end(); ++config) {
		double product = 1.0;
		for(std::vector<unsigned int>::const_iterator var =
							config->begin();
		    var != config->end(); var++)
			product *= values[*var];

		iter(product);
	}
}

std::vector<double> ProcMultiply::deriv(
				ValueIterator iter, unsigned int n) const
{
	std::vector<double> values;
	std::vector<unsigned int> offsets;
	unsigned int size = 0;
	while(iter) {
		offsets.push_back(size);
		size += iter.size();
		values.push_back(*iter++);
	}

	std::vector<double> result(out.size() * size, 0.0);
	unsigned int k = 0;
	for(std::vector<Config>::const_iterator config = out.begin();
	    config != out.end(); ++config, k++) {
		for(unsigned int i = 0; i < config->size(); i++) {
			double product = 1.0;
			for(unsigned int j = 0; j < config->size(); j++)
				if (i != j)
					product *= values[(*config)[j]];

			result[k * size + offsets[i]] = product;
		}
	}

	return result;
}

} // anonymous namespace

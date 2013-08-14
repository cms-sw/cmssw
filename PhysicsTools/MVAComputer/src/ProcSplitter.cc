// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcSplitter
// 

// Implementation:
//     Splits the first n instances of the input variables into separate
//     output variables.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcSplitter.cc,v 1.4 2009/06/03 09:50:14 saout Exp $
//

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcSplitter : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcSplitter,
					Calibration::ProcSplitter> Registry;

	ProcSplitter(const char *name,
	             const Calibration::ProcSplitter *calib,
	             const MVAComputer *computer);
	virtual ~ProcSplitter() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;
	virtual std::vector<double> deriv(
				ValueIterator iter, unsigned int n) const;

    private:
	unsigned int	count;
};

static ProcSplitter::Registry registry("ProcSplitter");

ProcSplitter::ProcSplitter(const char *name,
                          const Calibration::ProcSplitter *calib,
                          const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	count(calib->nFirst)
{
}

void ProcSplitter::configure(ConfIterator iter, unsigned int n)
{
	while(iter) {
		for(unsigned int i = 0; i < count; i++)
			iter << Variable::FLAG_OPTIONAL;
		iter << iter++(Variable::FLAG_ALL);
	}
}

void ProcSplitter::eval(ValueIterator iter, unsigned int n) const
{
	while(iter) {
		unsigned int i = 0;
		while(i < iter.size()) {
			iter << iter[i];
			if (i++ < count)
				iter();
		}
		while(i++ < count)
			iter();
		iter();
		iter++;
	}
}

std::vector<double> ProcSplitter::deriv(
				ValueIterator iter, unsigned int n) const
{
	unsigned int size = 0;
	for(ValueIterator iter2 = iter; iter2; ++iter2)
		size += iter2.size();

	std::vector<double> result(size * size);
	for(unsigned int i = 0; i < size; i++)
		result[i * size + i] = 1.0;

	return result;
}

} // anonymous namespace

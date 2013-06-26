// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcClassed
// 

// Implementation:
//     Variable processor that splits an input variable into n output
//     variables depending on the integer value of the input variable.
//     If the input variable has the value n, the nth output variable
//     is set to 1, all others to 0.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcClassed.cc,v 1.4 2010/02/20 12:16:20 saout Exp $
//

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcClassed : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcClassed,
					Calibration::ProcClassed> Registry;

	ProcClassed(const char *name,
	            const Calibration::ProcClassed *calib,
	            const MVAComputer *computer);
	virtual ~ProcClassed() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;

    private:
	unsigned int	nClasses;
};

static ProcClassed::Registry registry("ProcClassed");

ProcClassed::ProcClassed(const char *name,
                         const Calibration::ProcClassed *calib,
                         const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	nClasses(calib->nClasses)
{
}

void ProcClassed::configure(ConfIterator iter, unsigned int n)
{
	if (n != 1)
		return;

	iter(Variable::FLAG_NONE);
	for(unsigned int i = 0; i < nClasses; i++)
		iter << Variable::FLAG_NONE;
}

void ProcClassed::eval(ValueIterator iter, unsigned int n) const
{
	unsigned int value = (unsigned int)(*iter + 0.5);

	for(unsigned int i = 0; i < nClasses; i++)
		iter(i == value ? 1.0 : 0.0);
}

} // anonymous namespace

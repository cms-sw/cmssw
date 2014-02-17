// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcCount
// 

// Implementation:
//     Variable processor that returns the number of input variable instances
//
// Author:      Christophe Saout
// Created:     Fri May 18 20:05 CEST 2007
// $Id: ProcCount.cc,v 1.2 2007/07/15 22:31:46 saout Exp $
//

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcCount : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcCount,
					Calibration::ProcCount> Registry;

	ProcCount(const char *name,
	          const Calibration::ProcCount *calib,
	          const MVAComputer *computer);
	virtual ~ProcCount() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;
};

static ProcCount::Registry registry("ProcCount");

ProcCount::ProcCount(const char *name,
                          const Calibration::ProcCount *calib,
                          const MVAComputer *computer) :
	VarProcessor(name, calib, computer)
{
}

void ProcCount::configure(ConfIterator iter, unsigned int n)
{
	while(iter)
		iter++(Variable::FLAG_ALL) << Variable::FLAG_NONE;
}

void ProcCount::eval(ValueIterator iter, unsigned int n) const
{
	while(iter) {
		unsigned int count = iter.size();
		iter(count);
		iter++;
	}
}

} // anonymous namespace

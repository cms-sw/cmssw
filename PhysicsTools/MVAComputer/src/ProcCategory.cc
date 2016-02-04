// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcCategory
// 

// Implementation:
//     Categorizes the input variables given a set of ranges for each
//     input variable. Output is an integer number.
//
// Author:      Christophe Saout
// Created:     Sun Sep 16 04:05 CEST 2007
// $Id: ProcCategory.cc,v 1.3 2007/10/21 14:49:46 saout Exp $
//

#include <algorithm>
#include <vector>

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcCategory : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcCategory,
					Calibration::ProcCategory> Registry;

	ProcCategory(const char *name,
	             const Calibration::ProcCategory *calib,
	             const MVAComputer *computer);
	virtual ~ProcCategory() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;

    private:
	typedef Calibration::ProcCategory::BinLimits BinLimits;

	const Calibration::ProcCategory	*calib;
};

static ProcCategory::Registry registry("ProcCategory");

ProcCategory::ProcCategory(const char *name,
                           const Calibration::ProcCategory *calib,
                           const MVAComputer *computer) :
	VarProcessor(name, calib, computer), calib(calib)
{
}

void ProcCategory::configure(ConfIterator iter, unsigned int n)
{
	if (n != calib->variableBinLimits.size())
		return;

	unsigned int categories = 1;
	for(std::vector<BinLimits>::const_iterator bin =
					calib->variableBinLimits.begin();
	    bin != calib->variableBinLimits.end(); bin++)
		categories *= (bin->size() + 1);

	if (calib->categoryMapping.size() != categories)
		return;

	while(iter)
		iter++(Variable::FLAG_NONE);

	iter << Variable::FLAG_NONE;
}

void ProcCategory::eval(ValueIterator iter, unsigned int n) const
{
	unsigned int category = 0;
	for(std::vector<BinLimits>::const_iterator vars =
					calib->variableBinLimits.begin();
	    vars != calib->variableBinLimits.end(); vars++, ++iter) {
		unsigned int idx = std::upper_bound(vars->begin(), vars->end(),
		                                    *iter) - vars->begin();
		category *= vars->size() + 1;
		category += idx;
	}

	iter(calib->categoryMapping[category]);
}

} // anonymous namespace

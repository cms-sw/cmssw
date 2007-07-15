// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcNormalize
// 

// Implementation:
//     Normalizes the input variables. n values in each input variable
//     is normalized to n values for each input variables. The normalization
//     consists of a range normalization step (min...max) and mapping step
//     that equalizes using the probability distribution (via PDF).
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcNormalize.cc,v 1.3 2007/05/17 15:04:08 saout Exp $
//

#include <algorithm>
#include <iterator>
#include <vector>

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/Spline.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcNormalize : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcNormalize,
					Calibration::ProcNormalize> Registry;

	ProcNormalize(const char *name,
	              const Calibration::ProcNormalize *calib,
	              const MVAComputer *computer);
	virtual ~ProcNormalize() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;

    private:
	struct Map {
		inline Map(const Calibration::PDF &pdf) :
			min(pdf.range.first),
			width(pdf.range.second - pdf.range.first),
			spline(pdf.distr.size(), &pdf.distr.front()) {}

		double		min, width;
		Spline		spline;
	};

	std::vector<Map>	maps;
};

static ProcNormalize::Registry registry("ProcNormalize");

ProcNormalize::ProcNormalize(const char *name,
                             const Calibration::ProcNormalize *calib,
                             const MVAComputer *computer) :
	VarProcessor(name, calib, computer)
{
	std::copy(calib->distr.begin(), calib->distr.end(),
	          std::back_inserter(maps));
}

void ProcNormalize::configure(ConfIterator iter, unsigned int n)
{
	if (n != maps.size())
		return;

	while(iter)
		iter << iter++(Variable::FLAG_ALL);
}

void ProcNormalize::eval(ValueIterator iter, unsigned int n) const
{
	for(std::vector<Map>::const_iterator map = maps.begin();
	    map != maps.end(); map++, ++iter) {
		for(double *value = iter.begin();
		    value < iter.end(); value++) {
			double val = *value;
			val = (val - map->min) / map->width;
			val = map->spline.integral(val);
			iter << val;
		}
		iter();
	}
}

} // anonymous namespace

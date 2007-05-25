// -*- C++ -*-
//
// Package:     Discriminator
// Class  :     ProcLikelihood
// 

// Implementation:
//     A likelihood estimator variable processor. Reads in 0..n values for
//     m variables and calculates the total signal/background likelihood
//     using calibration PDFs for signal and background for each variable.
//     The output variable is set to s/(s+b).
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcLikelihood.cc,v 1.2 2007/05/17 15:04:08 saout Exp $
//

#include <algorithm>
#include <iterator>
#include <vector>

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/Spline.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcLikelihood : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcLikelihood,
					Calibration::ProcLikelihood> Registry;

	ProcLikelihood(const char *name,
	               const Calibration::ProcLikelihood *calib,
	               const MVAComputer *computer);
	virtual ~ProcLikelihood() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;

    private:
	struct PDF {
		PDF(const Calibration::PDF *calib) :
			min(calib->range.first),
			width(calib->range.second - calib->range.first),
			spline(calib->distr.size(), &calib->distr.front()) {}

		inline double eval(double value) const;

		double		min, width;
		Spline		spline;
	};

	struct SigBkg {
		SigBkg(const Calibration::ProcLikelihood::SigBkg &calib) :
			signal(&calib.signal),
			background(&calib.background) {}

		PDF	signal;
		PDF	background;
	};

	std::vector<SigBkg>	pdfs;
};

static ProcLikelihood::Registry registry("ProcLikelihood");

inline double ProcLikelihood::PDF::eval(double value) const
{
	value = (value - min) / width;
	return spline.eval(value) / spline.getArea();
}

ProcLikelihood::ProcLikelihood(const char *name,
                               const Calibration::ProcLikelihood *calib,
                               const MVAComputer *computer) :
	VarProcessor(name, calib, computer)
{
	std::copy(calib->pdfs.begin(), calib->pdfs.end(),
	          std::back_inserter(pdfs));
}

void ProcLikelihood::configure(ConfIterator iter, unsigned int n)
{
	if (n != pdfs.size())
		return;

	while(iter)
		iter++(Variable::FLAG_ALL);

	iter << Variable::FLAG_OPTIONAL;
}

void ProcLikelihood::eval(ValueIterator iter, unsigned int n) const
{
	bool empty = true;
	double signal = 1.0;
	double background = 1.0;

	for(std::vector<SigBkg>::const_iterator pdf = pdfs.begin();
	    pdf != pdfs.end(); pdf++, ++iter) {
		for(double *value = iter.begin();
		    value < iter.end(); value++) {
			empty = false;
			double signalProb = pdf->signal.eval(*value);
			double backgroundProb = pdf->background.eval(*value);
			signal *= std::max(0.0, signalProb);
			background *= std::max(0.0, backgroundProb);
		}
	}

	if (empty)
		iter();
	else if (signal + background > 1.0e-20)
		iter(signal / (signal + background));
	else
		iter(0.5);
}

} // anonymous namespace

// -*- C++ -*-
//
// Package:     MVAComputer
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
// $Id: ProcLikelihood.cc,v 1.5 2007/09/16 22:55:34 saout Exp $
//

#include <vector>
#include <memory>

#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"  
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
		virtual ~PDF() {}
		virtual double eval(double value) const = 0;
	};

	struct SplinePDF : public PDF {
		SplinePDF(const Calibration::HistogramF *calib) :
			min(calib->range().min),
			width(calib->range().width())
		{
			std::vector<double> values(
					calib->values().begin() + 1,
					calib->values().end() - 1);
			spline.set(values.size(), &values.front());
		}

		virtual double eval(double value) const;

		double		min, width;
		Spline		spline;
	};

	struct HistogramPDF : public PDF {
		HistogramPDF(const Calibration::HistogramF *calib) :
			histo(calib) {}

		virtual double eval(double value) const;

		const Calibration::HistogramF	*histo;
	};

	struct SigBkg {
		SigBkg(const Calibration::ProcLikelihood::SigBkg &calib)
		{
			if (calib.useSplines) {
				signal = std::auto_ptr<PDF>(
					new SplinePDF(&calib.signal));
				background = std::auto_ptr<PDF>(
					new SplinePDF(&calib.background));
			} else {
				signal = std::auto_ptr<PDF>(
					new HistogramPDF(&calib.signal));
				background = std::auto_ptr<PDF>(
					new HistogramPDF(&calib.background));
			}
		}

		std::auto_ptr<PDF>	signal;
		std::auto_ptr<PDF>	background;
	};

	std::vector<SigBkg>	pdfs;
	unsigned int		nCategories;
	double			bias;
};

static ProcLikelihood::Registry registry("ProcLikelihood");

double ProcLikelihood::SplinePDF::eval(double value) const
{
	value = (value - min) / width;
	return spline.eval(value) / spline.getArea();
}

double ProcLikelihood::HistogramPDF::eval(double value) const
{
	return histo->normalizedValue(value);
}

ProcLikelihood::ProcLikelihood(const char *name,
                               const Calibration::ProcLikelihood *calib,
                               const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	pdfs(calib->pdfs.begin(), calib->pdfs.end()),
	nCategories(calib->nCategories),
	bias(calib->bias)
{
}

void ProcLikelihood::configure(ConfIterator iter, unsigned int n)
{
	if (nCategories) {
		if (n < 1 || (n - 1) * nCategories != pdfs.size())
			return;
		iter++(Variable::FLAG_NONE);
	} else if (n != pdfs.size())
		return;

	while(iter)
		iter++(Variable::FLAG_ALL);

	iter << Variable::FLAG_OPTIONAL;
}

void ProcLikelihood::eval(ValueIterator iter, unsigned int n) const
{
	std::vector<SigBkg>::const_iterator pdf;
	std::vector<SigBkg>::const_iterator last;

	if (nCategories) {
		int cat = (int)*iter++;
		if (cat < 0 || (unsigned int)cat >= nCategories) {
			iter();
			return;
		}

		pdf = pdfs.begin() + cat * (n - 1);
		last = pdf + (n - 1);
	} else {
		pdf = pdfs.begin();
		last = pdfs.end();
	}		

	bool empty = true;
	double signal = bias;
	double background = 1.0;

	for(; pdf != last; ++pdf, ++iter) {
		for(double *value = iter.begin();
		    value < iter.end(); value++) {
			empty = false;
			double signalProb = pdf->signal->eval(*value);
			double backgroundProb = pdf->background->eval(*value);
			signal *= std::max(0.0, signalProb);
			background *= std::max(0.0, backgroundProb);
		}
	}

	if (empty || signal + background < 1.0e-30)
		iter();
	else
		iter(signal / (signal + background));
}

} // anonymous namespace

// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcLikelihood
// 

// Implementation:
//     A likelihood estimator variable processor. Reads in 0..n values for
//     m variables and calculates the total signal/background likelihood
//     using calibration PDFs for signal and background for each variable.
//     The output variable is set to s/(s+b) (or log(s/b) for logOutput).
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcLikelihood.cc,v 1.9 2008/04/21 08:53:03 saout Exp $
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

		double	norm;
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
			double norm = (calib.signal.numberOfBins() +
			               calib.background.numberOfBins()) / 2.0;
			signal->norm = norm;
			background->norm = norm;
		}

		std::auto_ptr<PDF>	signal;
		std::auto_ptr<PDF>	background;
	};

	std::vector<SigBkg>	pdfs;
	std::vector<double>	bias;
	int			categoryIdx;
	bool			logOutput;
	bool			individual;
	bool			neverUndefined;
	bool			keepEmpty;
	unsigned int		nCategories;
};

static ProcLikelihood::Registry registry("ProcLikelihood");

double ProcLikelihood::SplinePDF::eval(double value) const
{
	value = (value - min) / width;
	return spline.eval(value) * norm / spline.getArea();
}

double ProcLikelihood::HistogramPDF::eval(double value) const
{
	return histo->normalizedValue(value) * norm;
}

ProcLikelihood::ProcLikelihood(const char *name,
                               const Calibration::ProcLikelihood *calib,
                               const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	pdfs(calib->pdfs.begin(), calib->pdfs.end()),
	bias(calib->bias),
	categoryIdx(calib->categoryIdx),
	nCategories(1)
{
	typedef PhysicsTools::Calibration::ProcLikelihood Calib;

	logOutput = (categoryIdx & (1 << Calib::kLogOutput)) != 0;
	individual = (categoryIdx & (1 << Calib::kLogOutput)) != 0;
	neverUndefined =
		(categoryIdx & (1 << Calib::kNeverUndefined)) != 0;
	keepEmpty = (categoryIdx & (1 << Calib::kKeepEmpty)) != 0;

	if (categoryIdx < 0)
		categoryIdx |= ~(int)((1 << (Calib::kCategoryMax + 1)) - 1);
	else
		categoryIdx &= (1 << (Calib::kCategoryMax + 1)) - 1;
}

void ProcLikelihood::configure(ConfIterator iter, unsigned int n)
{
	if (categoryIdx >= 0) {
		if ((int)n < categoryIdx + 1)
			return;
		nCategories = pdfs.size() / (n - 1);
		if (nCategories * (n - 1) != pdfs.size())
			return;
		if (!bias.empty() && bias.size() != nCategories)
			return;
	} else if (n != pdfs.size() || bias.size() > 1)
		return;

	int i = 0;
	while(iter) {
		if (categoryIdx == i++)
			iter++(Variable::FLAG_NONE);
		else
			iter++(Variable::FLAG_ALL);
	}

	if (individual) {
		for(unsigned int i = 0; i < pdfs.size(); i++)
			iter << (neverUndefined ? Variable::FLAG_NONE
			                        : Variable::FLAG_OPTIONAL);
	} else
		iter << (neverUndefined ? Variable::FLAG_NONE
		                        : Variable::FLAG_OPTIONAL);
}

void ProcLikelihood::eval(ValueIterator iter, unsigned int n) const
{
	std::vector<SigBkg>::const_iterator pdf;
	std::vector<SigBkg>::const_iterator last;

	int cat;
	if (categoryIdx >= 0) {
		ValueIterator iter2 = iter;
		for(int i = 0; i < categoryIdx; i++)
			++iter2;

		cat = (int)*iter2;
		if (cat < 0 || (unsigned int)cat >= nCategories) {
			iter();
			return;
		}

		pdf = pdfs.begin() + cat * (n - 1);
		last = pdf + (n - 1);
	} else {
		cat = 0;
		pdf = pdfs.begin();
		last = pdfs.end();
	}		

	int vars = 0;
	long double signal = bias.empty() ? 1.0 : bias[cat];
	long double background = 1.0;

	for(int i = 0; pdf != last; ++iter, i++) {
		if (i == categoryIdx)
			continue;
		for(double *value = iter.begin();
		    value < iter.end(); value++) {
			double signalProb =
				std::max(0.0, pdf->signal->eval(*value));
			double backgroundProb =
				std::max(0.0, pdf->background->eval(*value));
			if (!keepEmpty && !individual &&
			    signalProb + backgroundProb < 1.0e-20)
				continue;
			vars++;

			if (individual) {
				signalProb *= signal;
				backgroundProb *= background;
				if (logOutput) {
					if (signalProb < 1.0e-9 &&
					    backgroundProb < 1.0e-9) {
						if (!neverUndefined)
							continue;
						iter << 0.0;
					} else if (signalProb < 1.0e-9)
						iter << -99999.0;
					else if (backgroundProb < 1.0e-9)
						iter << +99999.0;
					else
						iter << (std::log(signalProb) -
						         std::log(backgroundProb));
				} else
					iter << (signalProb /
					         (signalProb + backgroundProb));
			} else {
				signal *= signalProb;
				background *= backgroundProb;
			}
		}

		++pdf;
		if (individual)
			iter();
	}

	if (!individual) {
		if (!vars || signal + background < std::exp(-7 * vars - 3)) {
			if (neverUndefined)
				iter(logOutput ? 0.0 : 0.5);
			else
				iter();
		} else if (logOutput) {
			if (signal < 1.0e-9 && background < 1.0e-9) {
				if (neverUndefined)
					iter(0.0);
				else
					iter();
			}
			else if (signal < 1.0e-9)
				iter(-99999.0);
			else if (background < 1.0e-9)
				iter(+99999.0);
			else
				iter(std::log(signal) - std::log(background));
		} else
			iter(signal / (signal + background));
	}
}

} // anonymous namespace

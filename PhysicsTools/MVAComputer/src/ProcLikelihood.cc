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
// $Id: ProcLikelihood.cc,v 1.15 2010/01/26 19:40:04 saout Exp $
//

#include <vector>
#include <memory>
#include <cmath>

#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"  
#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/Spline.h"
#include "FWCore/Utilities/interface/isFinite.h"

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
	virtual std::vector<double> deriv(
				ValueIterator iter, unsigned int n) const;

    private:
	struct PDF {
		virtual ~PDF() {}
		virtual double eval(double value) const = 0;
		virtual double deriv(double value) const = 0;

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
		virtual double deriv(double value) const;

		double		min, width;
		Spline		spline;
	};

	struct HistogramPDF : public PDF {
		HistogramPDF(const Calibration::HistogramF *calib) :
			histo(calib) {}

		virtual double eval(double value) const;
		virtual double deriv(double value) const;

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

	int findPDFs(ValueIterator iter, unsigned int n,
	             std::vector<SigBkg>::const_iterator &begin,
	             std::vector<SigBkg>::const_iterator &end) const;

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

double ProcLikelihood::SplinePDF::deriv(double value) const
{
	value = (value - min) / width;
	return spline.deriv(value) * norm / spline.getArea();
}

double ProcLikelihood::HistogramPDF::eval(double value) const
{
	return histo->normalizedValue(value) * norm;
}

double ProcLikelihood::HistogramPDF::deriv(double value) const
{
	return 0;
}

ProcLikelihood::ProcLikelihood(const char *name,
                               const Calibration::ProcLikelihood *calib,
                               const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	pdfs(calib->pdfs.begin(), calib->pdfs.end()),
	bias(calib->bias),
	categoryIdx(calib->categoryIdx), logOutput(calib->logOutput),
	individual(calib->individual), neverUndefined(calib->neverUndefined),
	keepEmpty(calib->keepEmpty), nCategories(1)
{
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
		for(unsigned int i = 0; i < pdfs.size(); i += nCategories)
			iter << (neverUndefined ? Variable::FLAG_NONE
			                        : Variable::FLAG_OPTIONAL);
	} else
		iter << (neverUndefined ? Variable::FLAG_NONE
		                        : Variable::FLAG_OPTIONAL);
}

int ProcLikelihood::findPDFs(ValueIterator iter, unsigned int n,
                             std::vector<SigBkg>::const_iterator &begin,
                             std::vector<SigBkg>::const_iterator &end) const
{
	int cat;
	if (categoryIdx >= 0) {
		ValueIterator iter2 = iter;
		for(int i = 0; i < categoryIdx; i++)
			++iter2;

		cat = (int)*iter2;
		if (cat < 0 || (unsigned int)cat >= nCategories)
			return -1;

		begin = pdfs.begin() + cat * (n - 1);
		end = begin + (n - 1);
	} else {
		cat = 0;
		begin = pdfs.begin();
		end = pdfs.end();
	}		

	return cat;
}

void ProcLikelihood::eval(ValueIterator iter, unsigned int n) const
{
	std::vector<SigBkg>::const_iterator pdf, last;
	int cat = findPDFs(iter, n, pdf, last);
	int vars = 0;
	long double signal = bias.empty() ? 1.0 : bias[cat];
	long double background = 1.0;

	if (cat < 0) {
		if (individual)
			for(unsigned int i = 0; i < n; i++)
				iter();
		else
			iter();
		return;
	}

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
				} else {
					double sum =
						signalProb + backgroundProb;
					if (sum > 1.0e-9)
						iter << (signalProb / sum);
					else if (neverUndefined)
						iter << 0.5;
				}
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
			} else if (signal < 1.0e-9)
				iter(-99999.0);
			else if (background < 1.0e-9)
				iter(+99999.0);
			else
				iter(std::log(signal) - std::log(background));
		} else
			iter(signal / (signal + background));
	}
}

std::vector<double> ProcLikelihood::deriv(ValueIterator iter,
                                          unsigned int n) const
{
	std::vector<SigBkg>::const_iterator pdf, last;
	int cat = findPDFs(iter, n, pdf, last);
	int vars = 0;
	long double signal = bias.empty() ? 1.0 : bias[cat];
	long double background = 1.0;

	std::vector<double> result;
	if (cat < 0)
		return result;

	unsigned int size = 0;
	for(ValueIterator iter2 = iter; iter2; ++iter2)
		size += iter2.size();

	// The logic whether a variable is used or net depends on the
	// evaluation, so FFS copy the whole ****

	if (!individual)
		result.resize(size);

	unsigned int j = 0;
	for(int i = 0; pdf != last; ++iter, i++) {
		if (i == categoryIdx) {
			j += iter.size();
			continue;
		}

		for(double *value = iter.begin();
		    value < iter.end(); value++, j++) {
			double signalProb = pdf->signal->eval(*value);
			double signalDiff = pdf->signal->deriv(*value);
			if (signalProb < 0.0)
				signalProb = signalDiff = 0.0;

			double backgroundProb = pdf->background->eval(*value);
			double backgroundDiff = pdf->background->deriv(*value);
			if (backgroundProb < 0.0)
				backgroundProb = backgroundDiff = 0.0;

			if (!keepEmpty && !individual &&
			    signalProb + backgroundProb < 1.0e-20)
				continue;
			vars++;

			if (individual) {
				signalProb *= signal;
				signalDiff *= signal;
				backgroundProb *= background;
				backgroundDiff *= background;
				if (logOutput) {
					if (signalProb < 1.0e-9 &&
					    backgroundProb < 1.0e-9) {
						if (!neverUndefined)
							continue;
						result.resize(result.size() +
						              size);
					} else if (signalProb < 1.0e-9 ||
					           backgroundProb < 1.0e-9)
						result.resize(result.size() +
						              size);
					else {
						result.resize(result.size() +
						              size);
						result[result.size() -
						       size + j] =
							signalDiff /
								signalProb -
							backgroundDiff /
								backgroundProb;
					}
				} else {
					double sum =
						signalProb + backgroundProb;
					if (sum > 1.0e-9) {
						result.resize(result.size() +
						              size);
						result[result.size() -
						       size + j] =
							(signalDiff *
							 backgroundProb -
							 signalProb *
							 backgroundDiff) /
							(sum * sum);
					} else if (neverUndefined)
						result.resize(result.size() +
						              size);
				}
			} else {
				signal *= signalProb;
				background *= backgroundProb;
				double s = signalDiff / signalProb;
				if (edm::isNotFinite(s))
					s = 0.0;
				double b = backgroundDiff / backgroundProb;
				if (edm::isNotFinite(b))
					b = 0.0;

				result[j] = s - b;
			}
		}

		++pdf;
	}

	if (!individual) {
		if (!vars || signal + background < std::exp(-7 * vars - 3)) {
			if (neverUndefined)
				std::fill(result.begin(), result.end(), 0.0);
			else
				result.clear();
		} else if (logOutput) {
			if (signal < 1.0e-9 && background < 1.0e-9) {
				if (neverUndefined)
					std::fill(result.begin(),
					          result.end(), 0.0);
				else
					result.clear();
			} else if (signal < 1.0e-9 ||
			           background < 1.0e-9)
				std::fill(result.begin(), result.end(), 0.0);
			else {
				// should be ok
			}
		} else {
			double factor = signal * background /
			                ((signal + background) *
			                 (signal + background));
			for(std::vector<double>::iterator p = result.begin();
			    p != result.end(); ++p)
				*p *= factor;
		}
	}

	return result;
}

} // anonymous namespace

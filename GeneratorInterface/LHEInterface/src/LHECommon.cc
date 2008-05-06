#include <algorithm>
#include <iostream>
#include <string>
#include <cctype>
#include <vector>
#include <cmath>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommon.h"

static int skipWhitespace(std::istream &in)
{
	int ch;
	do {
		ch = in.get();
	} while(std::isspace(ch));
	if (ch != std::istream::traits_type::eof())
		in.putback(ch);
	return ch;
}

namespace lhef {

LHECommon::LHECommon(std::istream &in)
{
	in >> heprup.IDBMUP.first >> heprup.IDBMUP.second
	   >> heprup.EBMUP.first >> heprup.EBMUP.second
	   >> heprup.PDFGUP.first >> heprup.PDFGUP.second
	   >> heprup.PDFSUP.first >> heprup.PDFSUP.second
	   >> heprup.IDWTUP >> heprup.NPRUP;
	if (!in.good())
		throw cms::Exception("InvalidFormat")
			<< "Les Houches file contained invalid"
			   " header in init section." << std::endl;

	heprup.resize();

	for(int i = 0; i < heprup.NPRUP; i++) {
		in >> heprup.XSECUP[i] >> heprup.XERRUP[i]
		   >> heprup.XMAXUP[i] >> heprup.LPRUP[i];
		if (!in.good())
			throw cms::Exception("InvalidFormat")
				<< "Les Houches file contained invalid data"
				   " in header payload line " << (i + 1)
				<< "." << std::endl;
	}

	skipWhitespace(in);
	if (!in.eof())
		edm::LogWarning("Generator|LHEInterface")
			<< "Les Houches file contained spurious"
			   " content after the regular data." << std::endl;

	init();
}

LHECommon::LHECommon(const HEPRUP &heprup) :
	heprup(heprup)
{
	init();
}

LHECommon::~LHECommon()
{
}

void LHECommon::init()
{
	for(int i = 0; i < heprup.NPRUP; i++) {
		Process proc;

		proc.process = heprup.LPRUP[i];
		proc.heprupIndex = (unsigned int)i;

		processes.push_back(proc);
	}

	std::sort(processes.begin(), processes.end());
}

bool LHECommon::operator == (const LHECommon &other) const
{
	return heprup == other.heprup;
}

void LHECommon::count(int process, CountMode mode, double weight)
{
	std::vector<Process>::iterator proc =
		std::lower_bound(processes.begin(), processes.end(), process);
	if (proc == processes.end() || proc->process != process)
		return;

	switch(mode) {
	    case kAccepted:
		proc->accepted.add(weight);
	    case kSelected:
		proc->selected.add(weight);
	    case kTried:
		proc->tried.add(weight);
	}
}

LHECommon::XSec LHECommon::xsec() const
{
	double sigSelSum = 0.0;
	double sigSum = 0.0;
	double err2Sum = 0.0;

	for(std::vector<Process>::const_iterator proc = processes.begin();
	    proc != processes.end(); ++proc) {
		unsigned int idx = proc->heprupIndex;

		double sigmaSum, sigma2Sum, sigma2Err;
		if (std::abs(heprup.IDWTUP == 3)) {
			sigmaSum = proc->tried.n * heprup.XSECUP[idx];
			sigma2Sum = sigmaSum * heprup.XSECUP[idx];
			sigma2Err = proc->tried.n * heprup.XERRUP[idx]
			                          * heprup.XERRUP[idx];
		} else {
			sigmaSum = proc->tried.sum;
			sigma2Sum = proc->tried.sum2;
			sigma2Err = 0.0;
		}

		if (!proc->accepted.n)
			continue;

		double sigmaAvg = sigmaSum / proc->tried.n;
		double fracAcc = (double)proc->accepted.n / proc->selected.n;
		double sigmaFin = sigmaAvg * fracAcc;

		double deltaFin = sigmaFin;
		if (proc->accepted.n > 1) {
			double sigmaAvg2 = sigmaAvg * sigmaAvg;
			double delta2Sig =
				(sigma2Sum / proc->tried.n - sigmaAvg2) /
				(proc->tried.n * sigmaAvg2);
			double delta2Veto =
				((double)proc->selected.n - proc->accepted.n) /
				((double)proc->selected.n * proc->accepted.n);
			double delta2Sum = delta2Sig + delta2Veto
			                   + sigma2Err / sigmaSum;
			deltaFin = sigmaFin * (delta2Sum > 0.0 ?
						std::sqrt(delta2Sum) : 0.0);
		}

		sigSelSum += sigmaAvg;
		sigSum += sigmaFin;
		err2Sum += deltaFin * deltaFin;
	}

	XSec result;
	result.value = 1.0e-9 * sigSum;
	result.error = 1.0e-9 * std::sqrt(err2Sum);

	return result;
}

} // namespace lhef

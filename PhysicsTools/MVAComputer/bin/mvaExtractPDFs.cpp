#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include <TFile.h>
#include <TH1.h>


#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/Spline.h"

using namespace PhysicsTools;

static const int precision = 100;

int main(int argc, char **argv)
{
	using Calibration::HistogramF;
	using Calibration::VarProcessor;

	if (argc != 3) {
		std::cerr << "Syntax: " << argv[0] << " <MVA File> "
		          << "<output ROOT file>" << std::endl;
		return 1;
	}

	Calibration::MVAComputer *calib =
		MVAComputer::readCalibration(argv[1]);
	if (!calib)
		return 1;

	std::map<std::string, HistogramF*> histos;

	std::vector<VarProcessor*> procs = calib->getProcessors();
	for(unsigned int z = 0; z < procs.size(); ++z) {
		VarProcessor *proc = procs[z];
		if (!proc)
			continue;

		std::ostringstream ss3;
		ss3 << (z + 1);

		Calibration::ProcLikelihood *lkh =
			dynamic_cast<Calibration::ProcLikelihood*>(proc);
		Calibration::ProcNormalize *norm =
			dynamic_cast<Calibration::ProcNormalize*>(proc);

		if (lkh) {
			for(unsigned int i = 0; i < lkh->pdfs.size(); i++) {
				std::ostringstream ss2;
				ss2 << (i + 1);
				histos["proc" + ss3.str() + "_sig" + ss2.str()] = &lkh->pdfs[i].signal;
				histos["proc" + ss3.str() + "_bkg" + ss2.str()] = &lkh->pdfs[i].background;
			}
		} else if (norm) {
			for(unsigned int i = 0; i < norm->distr.size(); i++) {
				std::ostringstream ss2;
				ss2 << (i + 1);
				histos["proc" + ss3.str() + "_norm" + ss2.str()] = &norm->distr[i];
			}
		}
	}

	TFile *f = TFile::Open(argv[2], "RECREATE");
	if (!f)
		return 2;

	for(std::map<std::string, HistogramF*>::const_iterator iter = histos.begin();
	    iter != histos.end(); ++iter) {
		std::string name = iter->first;
		HistogramF *histo = iter->second;

		unsigned int size = histo->values().size() - 2;
		std::vector<double> values(
				histo->values().begin() + 1,
				histo->values().end() - 1);
		Spline spline;
		spline.set(values.size(), &values.front());

		double min = histo->range().min;
		double max = histo->range().max;

		TH1F *h = new TH1F((name + "_histo").c_str(), (name + "_histo").c_str(),
		                   size, min - 0.5 * (max - min) / size,
		                   max + 0.5 * (max - min) / size);
		TH1F *s = new TH1F((name + "_spline").c_str(), (name + "_spline").c_str(),
		                   size * precision, min, max);

		for(unsigned int i = 0; i < size; i++) {
			h->SetBinContent(i + 1, histo->values()[i + 1]);
			for(int j = 0; j < precision; j++) {
				unsigned int k = i * precision + j;
				double x = (k + 0.5) / (size * precision);
				double v = spline.eval(x);
				s->SetBinContent(k, v);
			}
		}
	}

	f->Write();
	delete f;

	return 0;
}

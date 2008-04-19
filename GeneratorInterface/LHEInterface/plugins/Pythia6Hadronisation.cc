#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <assert.h>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/PythiaWrapper6_2.h>
#include <HepMC/IO_HEPEVT.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommonBlocks.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"

namespace lhef {

class Pythia6Hadronisation : public Hadronisation {
    public:
	Pythia6Hadronisation(const edm::ParameterSet &params);
	~Pythia6Hadronisation();

	struct FortranCallback;

    protected:
	friend struct FortranCallback;

	void fillHeader();
	void fillEvent();
	bool veto();

    private:
	std::auto_ptr<HepMC::GenEvent> doHadronisation();
	void newCommon(const boost::shared_ptr<LHECommon> &common);

	const int		pythiaPylistVerbosity;
	int			maxEventsToPrint;
	int			iterations;

	HepMC::IO_HEPEVT	conv;
};

struct Pythia6Hadronisation::FortranCallback {
	FortranCallback() : instance(0) {}

	void upinit() { instance->fillHeader(); }
	void upevnt() { instance->fillEvent(); }
	bool upveto() { return instance->veto(); }

	lhef::Pythia6Hadronisation *instance;
} static fortranCallback;

extern "C" {
	void pygive_(const char *line, int length);

	static bool call_pygive(const std::string &line)
	{
		int numWarn = pydat1.mstu[26];	// # warnings
		int numErr = pydat1.mstu[22];	// # errors

		pygive_(line.c_str(), line.length());

		return pydat1.mstu[26] == numWarn &&
		       pydat1.mstu[22] == numErr;
	}

	void upinit_() { fortranCallback.upinit(); }
	void upevnt_() { fortranCallback.upevnt(); }
	void upveto_(int *veto) { *veto = fortranCallback.upveto(); }
} // extern "C"

Pythia6Hadronisation::Pythia6Hadronisation(const edm::ParameterSet &params) :
	Hadronisation(params),
	pythiaPylistVerbosity(params.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
	maxEventsToPrint(params.getUntrackedParameter<int>("maxEventsToPrint", 0))
{
	std::vector<std::string> setNames =
		params.getParameter<std::vector<std::string> >("parameterSets");

	for(std::vector<std::string>::const_iterator iter = setNames.begin();
	    iter != setNames.end(); ++iter) {
		std::vector<std::string> lines =
			params.getParameter< std::vector<std::string> >(*iter);

		edm::LogInfo("Generator|LHEInterface") << "----------------------------------------------";
		edm::LogInfo("Generator|LHEInterface") << "Read PYTHIA parameter set " << *iter;
		edm::LogInfo("Generator|LHEInterface") << "----------------------------------------------";

		for(std::vector<std::string>::const_iterator line = lines.begin();
		    line != lines.end(); ++line ) {
			if (line->substr(0, 7) == "MRPY(1)")
				throw cms::Exception("PythiaError")
					<< "Attempted to set random number"
					   " using Pythia command 'MRPY(1)'."
					   " Please use the"
					   " RandomNumberGeneratorService."
					<< std::endl;
			if (!call_pygive(*line))
				throw cms::Exception("PythiaError")
					<< "Pythia did not accept \""
					<< *line << "\"." << std::endl;
		}
	}

	edm::Service<edm::RandomNumberGenerator> rng;
	std::ostringstream ss;
	ss << "MRPY(1)=" << rng->mySeed();
	call_pygive(ss.str());
}

Pythia6Hadronisation::~Pythia6Hadronisation()
{
}

std::auto_ptr<HepMC::GenEvent> Pythia6Hadronisation::doHadronisation()
{
	iterations = 0;
	assert(!fortranCallback.instance);
	fortranCallback.instance = this;
	call_pyevnt();
	call_pyhepc(1);
	fortranCallback.instance = 0;

	if (iterations > 1)
		return std::auto_ptr<HepMC::GenEvent>();

	std::auto_ptr<HepMC::GenEvent> event(conv.read_next_event());

	getRawEvent()->fillEventInfo(event.get());
	if (event->event_scale() < 0.0)
		event->set_event_scale(pypars.pari[16]);
	if (event->alphaQED() < 0.0)
		event->set_alphaQED(pypars.pari[27]);
	if (event->alphaQCD() < 0.0)
		event->set_alphaQCD(pypars.pari[28]);

	HepMC::PdfInfo pdf;
	getRawEvent()->fillPdfInfo(&pdf);
	if (pdf.pdf1() < 0)
		pdf.set_pdf1(pypars.pari[28]);
	if (pdf.pdf2() < 0)
		pdf.set_pdf2(pypars.pari[29]);
	if (pdf.x1() < 0)
		pdf.set_x1(pypars.pari[32]);
	if (pdf.x2() < 0)
		pdf.set_x2(pypars.pari[33]);
	if (pdf.scalePDF() < 0)
		pdf.set_scalePDF(pypars.pari[20]);

std::cout << pdf.id1() << ", "
	<< pdf.id2() << ", "
	<< pdf.pdf1() << ", "
	<< pdf.pdf2() << ", "
	<< pdf.x1() << ", "
	<< pdf.x2() << ", "
	<< pdf.scalePDF() << std::endl;

	if (maxEventsToPrint > 0) {
		maxEventsToPrint--;
		if (pythiaPylistVerbosity)
			call_pylist(pythiaPylistVerbosity);
	}

	return event;
}

void Pythia6Hadronisation::newCommon(const boost::shared_ptr<LHECommon> &common)
{
	assert(!fortranCallback.instance);
	fortranCallback.instance = this;
	call_pyinit("USER", "", "", 0.0);
	fortranCallback.instance = 0;
}

void Pythia6Hadronisation::fillHeader()
{
	const HEPRUP *heprup = getRawEvent()->getHEPRUP();

	CommonBlocks::fillHEPRUP(heprup);
}

void Pythia6Hadronisation::fillEvent()
{
	const HEPEUP *hepeup = getRawEvent()->getHEPEUP();

	if (iterations++) {
		hepeup_.nup = 0;
		return;
	}

	CommonBlocks::fillHEPEUP(hepeup);
}

bool Pythia6Hadronisation::veto()
{
	return false;
}

DEFINE_LHE_HADRONISATION_PLUGIN(Pythia6Hadronisation);

} // namespace lhef

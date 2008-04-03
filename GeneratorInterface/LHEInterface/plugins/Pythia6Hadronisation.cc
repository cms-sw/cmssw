#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <assert.h>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PythiaWrapper6_2.h>
#include <HepMC/IO_HEPEVT.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
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
	double getCrossSection() const;
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

	extern struct HEPRUP_ {
		int idbmup[2];
		double ebmup[2];
		int pdfgup[2];
		int pdfsup[2];
		int idwtup;
		int nprup;
		double xsecup[100];
		double xerrup[100];
		double xmaxup[100];
		int lprup[100];
	} heprup_;

	extern struct HEPEUP_ {
		int nup;
		int idprup;
		double xwgtup;
		double scalup;
		double aqedup;
		double aqcdup;
		int idup[500];
		int istup[500];
		int mothup[500][2];
		int icolup[500][2];
		double pup[500][5];
		double vtimup[500];
		double spinup[500];
	} hepeup_;

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

	event->set_signal_process_id(pypars.msti[0]);   

	if (maxEventsToPrint > 0) {
		maxEventsToPrint--;
		if (pythiaPylistVerbosity)
			call_pylist(pythiaPylistVerbosity);
	}

	return event;
}

double Pythia6Hadronisation::getCrossSection() const
{
	return pypars.pari[0];
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

	heprup_.idbmup[0] = heprup->IDBMUP.first;
	heprup_.idbmup[1] = heprup->IDBMUP.second;
	heprup_.ebmup[0] = heprup->EBMUP.first;
	heprup_.ebmup[1] = heprup->EBMUP.second;
	heprup_.pdfgup[0] = heprup->PDFGUP.first;
	heprup_.pdfgup[1] = heprup->PDFGUP.second;
	heprup_.pdfsup[0] = heprup->PDFSUP.first;
	heprup_.pdfsup[1] = heprup->PDFSUP.second;
	heprup_.idwtup = heprup->IDWTUP;
	heprup_.nprup = heprup->NPRUP;
	for(int i = 0; i < heprup->NPRUP; i++) {
		heprup_.xsecup[i] = heprup->XSECUP[i];
		heprup_.xerrup[i] = heprup->XERRUP[i];
		heprup_.xmaxup[i] = heprup->XMAXUP[i];
		heprup_.lprup[i] = heprup->LPRUP[i];
	}
}

void Pythia6Hadronisation::fillEvent()
{
	const HEPEUP *hepeup = getRawEvent()->getHEPEUP();

	if (iterations++) {
		hepeup_.nup = 0;
		return;
	}

	hepeup_.nup = hepeup->NUP;
	hepeup_.idprup = hepeup->IDPRUP;
	hepeup_.xwgtup = hepeup->XWGTUP;
	hepeup_.scalup = hepeup->SCALUP;
	hepeup_.aqedup = hepeup->AQEDUP;
	hepeup_.aqcdup = hepeup->AQCDUP;
	for(int i = 0; i < hepeup->NUP; i++) {
		hepeup_.idup[i] = hepeup->IDUP[i];
		hepeup_.istup[i] = hepeup->ISTUP[i];
		hepeup_.mothup[i][0] = hepeup->MOTHUP[i].first;
		hepeup_.mothup[i][1] = hepeup->MOTHUP[i].second;
		hepeup_.icolup[i][0] = hepeup->ICOLUP[i].first;
		hepeup_.icolup[i][1] = hepeup->ICOLUP[i].second;
		for(unsigned int j = 0; j < 5; j++)
			hepeup_.pup[i][j] = hepeup->PUP[i][j];
		hepeup_.vtimup[i] = hepeup->VTIMUP[i];
		hepeup_.spinup[i] = hepeup->SPINUP[i];
	}
}

bool Pythia6Hadronisation::veto()
{
	return false;
}

DEFINE_LHE_HADRONISATION_PLUGIN(Pythia6Hadronisation);

} // namespace lhef

#include <iostream>
#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <assert.h>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/PythiaWrapper6_2.h>
#include <HepMC/HEPEVT_Wrapper.h>
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
	void doInit();
	std::auto_ptr<HepMC::GenEvent> doHadronisation();
	void newCommon(const boost::shared_ptr<LHECommon> &common);

	std::vector<std::string>	paramLines;

	const int			pythiaPylistVerbosity;
	int				maxEventsToPrint;
	int				iterations;

	HepMC::IO_HEPEVT		conv;
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

		for(std::vector<std::string>::const_iterator line = lines.begin();
		    line != lines.end(); ++line ) {
			if (line->substr(0, 7) == "MRPY(1)")
				throw cms::Exception("PythiaError")
					<< "Attempted to set random number"
					   " using Pythia command 'MRPY(1)'."
					   " Please use the"
					   " RandomNumberGeneratorService."
					<< std::endl;

			paramLines.push_back(*line);
		}
	}

	edm::Service<edm::RandomNumberGenerator> rng;
	std::ostringstream ss;
	ss << "MRPY(1)=" << rng->mySeed();
	paramLines.push_back(ss.str());
}

Pythia6Hadronisation::~Pythia6Hadronisation()
{
}

void Pythia6Hadronisation::doInit()
{
	for(std::vector<std::string>::const_iterator iter = paramLines.begin();
	    iter != paramLines.end(); ++iter) {
		if (!call_pygive(*iter))
			throw cms::Exception("PythiaError")
				<< "Pythia did not accept \""
				<< *iter << "\"." << std::endl;
	}

	call_pygive("MSEL=0");
	call_pygive(std::string("MSTP(143)=") +
	            (wantsShoweredEvent() ? "1" : "0"));
}

std::auto_ptr<HepMC::GenEvent> Pythia6Hadronisation::doHadronisation()
{
	iterations = 0;
	assert(!fortranCallback.instance);
	fortranCallback.instance = this;
	call_pyevnt();
	call_pyhepc(1);
	fortranCallback.instance = 0;

	if (iterations > 1 || hepeup_.nup <= 0 || pypars.msti[0] == 1)
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
	event->set_pdf_info(pdf);

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

namespace {
	struct SavedHEPEVT {
		int	index;
		int	id;
		int	mo1, mo2;

		void load(int index_)
		{
			index = index_;
			id = HepMC::HEPEVT_Wrapper::id(index);
			mo1 = HepMC::HEPEVT_Wrapper::first_parent(index);
			mo2 = HepMC::HEPEVT_Wrapper::last_parent(index);
		}

		void save()
		{
			HepMC::HEPEVT_Wrapper::set_id(index, id);
			HepMC::HEPEVT_Wrapper::set_parents(index, mo1, mo2);
		}

		SavedHEPEVT(int index) { load(index); }
	};
} // anonymous namespace

bool Pythia6Hadronisation::veto()
{
call_pylist(1);
	std::vector<SavedHEPEVT> saved;
	int n = HepMC::HEPEVT_Wrapper::number_entries();

	SavedHEPEVT i1(3), i2(4), p1(5), p2(6);
	saved.push_back(i1);
	saved.push_back(i2);
	i1.mo1 = i1.mo2 = 1, i1.id = p1.id, i1.save();
	i2.mo1 = i2.mo2 = 2, i2.id = p2.id, i2.save();

	for(int i = 7; i <= n; i++) {
		SavedHEPEVT p(i);
		if (p.mo1)
			break;

		saved.push_back(p);
		p.mo1 = 5, p.mo2 = 6;
		p.save();
	}

	// all particles added by pythia have status 1, so assume rest is 3
	boost::shared_ptr<HepMC::GenEvent> event(conv.read_next_event());

	// restore modified HepEVT content
	std::for_each(saved.begin(), saved.end(),
	              std::mem_fun_ref(&SavedHEPEVT::save));

	for(HepMC::GenEvent::particle_iterator iter = event->particles_begin();
	    iter != event->particles_end(); ++iter)
		if ((*iter)->status() == 2)
			(*iter)->set_status(3);


event->print();
	return showeredEvent(event);
}

DEFINE_LHE_HADRONISATION_PLUGIN(Pythia6Hadronisation);

} // namespace lhef

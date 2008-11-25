#include <algorithm>
#include <functional>
#include <iterator>
#include <iostream>
#include <sstream>
#include <fstream> 
#include <cstring>
#include <cstdio>
#include <cctype>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <assert.h>

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/PythiaWrapper6_2.h>
#include <HepMC/HEPEVT_Wrapper.h>
#include <HepMC/IO_HEPEVT.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "GeneratorInterface/CommonInterface/interface/TauolaInterface.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"

namespace lhef {

class Pythia6Hadronisation : public Hadronisation {
    public:
	Pythia6Hadronisation(const edm::ParameterSet &params);
	~Pythia6Hadronisation();

	struct FortranCallback;

	class Addon {
	    public:
		Addon() {}
		virtual ~Addon() {}

		virtual void init() {}
		virtual void beforeEvent() {}
		virtual void afterEvent() {}
		virtual void statistics() {}

		typedef boost::shared_ptr<Addon> Ptr;

		static Ptr create(const std::string &name,
		                  const edm::ParameterSet &params);
	};

    protected:
	friend struct FortranCallback;

	void fillHeader();
	void fillEvent();
	bool veto();

    private:
	void doInit();
	std::auto_ptr<HepMC::GenEvent> doHadronisation();
	void newRunInfo(const boost::shared_ptr<LHERunInfo> &runInfo);
	void statistics();
	double totalBranchingRatio(int pdgId) const;

	std::set<std::string> capabilities() const;

	std::vector<std::string>	paramLines;

	const int			pythiaPylistVerbosity;
	int				maxEventsToPrint;
	int				iterations;
	bool				vetoDone;

	std::vector<Addon::Ptr>		addons;

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
	void txgive_(const char *line, int length);
	void txgive_init_(void);

	int pycomp_(int *ip);
	void pyslha_(int *mupda, int *kforig, int *iretrn);

	void fioopn_(int *unit, const char *line, int length);
	void fiocls_(int *unit);

	extern struct PYDAT1 {
		int	mstu[200];
		double	paru[200];
		int	mstj[200];
		double	parj[200];
	} pydat1_;

	extern struct PYDAT2 {
		int	kchg[4][500];
		double	pmas[4][500];
		double	parf[2000];
		double	vckm[4][4];
	} pydat2_;

	extern struct PYINT4 {
		int	mwid[500];
		double	wids[5][500];
	} pyint4_;

	static bool call_pygive(const std::string &line)
	{
		int numWarn = pydat1.mstu[26];	// # warnings
		int numErr = pydat1.mstu[22];	// # errors

		pygive_(line.c_str(), line.length());

		return pydat1.mstu[26] == numWarn &&
		       pydat1.mstu[22] == numErr;
	}

	static bool call_txgive(const std::string &line)
	{
		txgive_(line.c_str(), line.length());
		return true;
	}

	static void call_txgive_init(void)
	{ txgive_init_(); }

	static int call_pyslha(int mupda, int kforig = 0)
	{
		int iretrn = 0;
		pyslha_(&mupda, &kforig, &iretrn);
		return iretrn;
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

	if (params.exists("externalGenerators")) {
		edm::ParameterSet pset =
			params.getParameter<edm::ParameterSet>(
							"externalGenerators");
		std::vector<std::string> externalGenerators =
			pset.getParameter<std::vector<std::string> >(
							"parameterSets");
		for(std::vector<std::string>::const_iterator iter =
						externalGenerators.begin();
		    iter != externalGenerators.end(); ++iter) {
			edm::ParameterSet generatorPSet =
				pset.getParameter<edm::ParameterSet>(*iter);
			Addon::Ptr addon = Addon::create(*iter, generatorPSet);
			addons.push_back(addon);
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

	call_txgive_init();

	std::for_each(addons.begin(), addons.end(),
	              boost::bind(&Addon::init, _1));
}

std::auto_ptr<HepMC::GenEvent> Pythia6Hadronisation::doHadronisation()
{
	iterations = 0;
	vetoDone = false;
	assert(!fortranCallback.instance);
	fortranCallback.instance = this;

	std::for_each(addons.begin(), addons.end(),
	              boost::bind(&Addon::beforeEvent, _1));

	call_pyevnt();

	if (iterations > 1 || hepeup_.nup <= 0 || pypars.msti[0] == 1) {
		fortranCallback.instance = 0;
		return std::auto_ptr<HepMC::GenEvent>();
	}

	std::for_each(addons.begin(), addons.end(),
	              boost::bind(&Addon::afterEvent, _1));

	call_pyhepc(1);

	fortranCallback.instance = 0;

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

static void processSLHA(const std::vector<std::string> &lines)
{
	std::set<std::string> blocks;
	unsigned int model = 0, subModel = 0;

	const char *fname = std::tmpnam(NULL);
	std::ofstream file(fname, std::fstream::out | std::fstream::trunc);
	std::string block;
	for(std::vector<std::string>::const_iterator iter = lines.begin();
	    iter != lines.end(); ++iter) {
		file << *iter;

		std::string line = *iter;
		std::transform(line.begin(), line.end(),
		               line.begin(), (int(*)(int))std::toupper);
		std::string::size_type pos = line.find('#');
		if (pos != std::string::npos)
			line.resize(pos);

		if (line.empty())
			continue;

		if (!boost::algorithm::is_space()(line[0])) {
			std::vector<std::string> tokens;
			boost::split(tokens, line,
			             boost::algorithm::is_space(),
			             boost::token_compress_on);
			if (!tokens.size())
				continue;
			block.clear();
			if (tokens.size() < 2)
				continue;
			if (tokens[0] == "BLOCK") {
				block = tokens[1];
				blocks.insert(block);
				continue;
			}

			if (tokens[0] == "DECAY") {
				block = "DECAY";
				blocks.insert(block);
			}
		} else if (block == "MODSEL") {
			std::istringstream ss(line);
			ss >> model >> subModel;
		} else if (block == "SMINPUTS") {
			std::istringstream ss(line);
			int index;
			double value;
			ss >> index >> value;
			switch(index) {
			    case 1:
				pydat1_.paru[103 - 1] = 1.0 / value;
				break;
			    case 2:
				pydat1_.paru[105 - 1] = value;
				break;
			    case 4:
				pydat2_.pmas[0][23 - 1] = value;
				break;
			    case 6:
				pydat2_.pmas[0][6 - 1] = value;
				break;
			    case 7:
				pydat2_.pmas[0][15 - 1] = value;
				break;
			}
		}
	}
	file.close();

	if (blocks.count("SMINPUTS"))
		pydat1_.paru[102 - 1] = 0.5 - std::sqrt(0.25 -
			pydat1_.paru[0] * M_SQRT1_2 *
			pydat1_.paru[103 - 1] /	pydat1_.paru[105 - 1] /
			(pydat2_.pmas[0][23 - 1] * pydat2_.pmas[0][23 - 1]));

	int unit = 24;
	fioopn_(&unit, fname, std::strlen(fname));
	std::remove(fname);

	call_pygive("IMSS(21)=24");
	call_pygive("IMSS(22)=24");

	if (model ||
	    blocks.count("HIGMIX") ||
	    blocks.count("SBOTMIX") ||
	    blocks.count("STOPMIX") ||
	    blocks.count("STAUMIX") ||
	    blocks.count("AMIX") ||
	    blocks.count("NMIX") ||
	    blocks.count("UMIX") ||
	    blocks.count("VMIX"))
		call_pyslha(1);
	if (model ||
	    blocks.count("QNUMBERS") ||
	    blocks.count("PARTICLE") ||
	    blocks.count("MINPAR") ||
	    blocks.count("EXTPAR") ||
	    blocks.count("SMINPUTS") ||
	    blocks.count("SMINPUTS"))
		call_pyslha(0);
	if (blocks.count("MASS"))
		call_pyslha(5, 0);
	if (blocks.count("DECAY"))
		call_pyslha(2);

	fiocls_(&unit);
}

void Pythia6Hadronisation::newRunInfo(
				const boost::shared_ptr<LHERunInfo> &runInfo)
{
	assert(!fortranCallback.instance);
	fortranCallback.instance = this;
	call_pyinit("USER", "", "", 0.0);
	fortranCallback.instance = 0;

	std::vector<std::string> slha = runInfo->findHeader("slha");
	if (!slha.empty()) {
		edm::LogInfo("Generator|LHEInterface")
			<< "Pythia6 hadronisation found an SLHA header, "
			<< "will be passed on to Pythia." << std::endl;
		processSLHA(slha);
	}
}

void Pythia6Hadronisation::statistics()
{
	std::for_each(addons.begin(), addons.end(),
	              boost::bind(&Addon::statistics, _1));
}

double Pythia6Hadronisation::totalBranchingRatio(int pdgId) const
{
	int pythiaId = pycomp_(&pdgId);
	return pyint4_.wids[2][pythiaId - 1];
}

std::set<std::string> Pythia6Hadronisation::capabilities() const
{
	std::set<std::string> result;
	result.insert("showeredEvent");
	result.insert("pythia6");
	result.insert("hepevt");
	return result;
}

void Pythia6Hadronisation::fillHeader()
{
	const HEPRUP *heprup = getRawEvent()->getHEPRUP();

	CommonBlocks::fillHEPRUP(heprup);

	onInit().emit();
}

void Pythia6Hadronisation::fillEvent()
{
	const HEPEUP *hepeup = getRawEvent()->getHEPEUP();

	if (iterations++) {
		hepeup_.nup = 0;
		return;
	}

	CommonBlocks::fillHEPEUP(hepeup);

	onBeforeHadronisation().emit();
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
	if (!hepeup_.nup || vetoDone) {
		edm::LogWarning("Generator|LHEInterface")
			<< "Pythia6 called UPVETO twice.  This usually "
			   "occurs after some internal error." << std::endl;
		return false;
	} else
		vetoDone = true;

	if (!wantsShoweredEventAsHepMC())
		return showeredEvent(boost::shared_ptr<HepMC::GenEvent>());

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

	return showeredEvent(event);
}

// handle addons

namespace {
	class TauolaAddon : public Pythia6Hadronisation::Addon {
	    public:
		TauolaAddon(const edm::ParameterSet &config) :
			usePolarization(config.getParameter<bool>("UseTauolaPolarization")),
			cards(config.getParameter<std::vector<std::string> >("InputCards"))
		{
		}

	    private:
		virtual void init()
		{
			if (usePolarization)
				tauola.enablePolarizationEffects();
			else
				tauola.disablePolarizationEffects();

			for(std::vector<std::string>::const_iterator iter =
				cards.begin(); iter != cards.end(); ++iter)
				call_txgive(*iter);

			tauola.initialize();
		}

		virtual void afterEvent() { tauola.processEvent(); }
		virtual void statistics() { tauola.print(); }

		bool				usePolarization;
		std::vector<std::string>	cards;

		edm::TauolaInterface		tauola;
	};
}  // anonymous namespace

Pythia6Hadronisation::Addon::Ptr
Pythia6Hadronisation::Addon::create(const std::string &name,
                                    const edm::ParameterSet &params)
{
	if (name == "Tauola")
		return Ptr(new TauolaAddon(params));
	else
		throw cms::Exception("Pythia6HadronisationError")
			<< "Pythia6 addon \"" << name << "\" unknown."
			<< std::endl;
}

DEFINE_LHE_HADRONISATION_PLUGIN(Pythia6Hadronisation);

} // namespace lhef

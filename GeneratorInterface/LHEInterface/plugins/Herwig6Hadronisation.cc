// Copyright notice:
// Large parts taken from Herwig6Interface, originally by Fabian Stoeckli

#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef _POSIX_C_SOURCE
#	include <sys/time.h>
#	include <signal.h>
#	include <setjmp.h>
#endif

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/HerwigWrapper6_4.h>
#include <HepMC/HEPEVT_Wrapper.h>
#include <HepMC/IO_HERWIG.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "GeneratorInterface/Herwig6Interface/src/herwig.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommonBlocks.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"

namespace lhef {

class Herwig6Hadronisation : public Hadronisation {
    public:
	Herwig6Hadronisation(const edm::ParameterSet &params);
	~Herwig6Hadronisation();

	struct FortranCallback;

    protected:
	friend struct FortranCallback;

	void fillHeader();
	void fillEvent();

    private:
	void clear();

	void doInit();
	std::auto_ptr<HepMC::GenEvent> doHadronisation();
	void newCommon(const boost::shared_ptr<LHECommon> &common);

	int				herwigVerbosity;
	int				maxEventsToPrint;
	bool				useJimmy;
	bool				doMPInteraction;
	int				numTrials;
	bool				builtinPDFs;
	std::string			lhapdfSetPath;
	std::vector<std::string>	paramLines;

	bool				needClear;
	HepMC::IO_HERWIG		conv;
	char				*buffer;
};

struct Herwig6Hadronisation::FortranCallback {
	FortranCallback() : instance(0) {}

	void upinit() { instance->fillHeader(); }
	void upevnt() { instance->fillEvent(); }

	lhef::Herwig6Hadronisation *instance;
} static fortranCallback;

extern "C" {
	void hwwarn_(const char *method, int *id);

	void setherwpdf_(void);
	void mysetpdfpath_(const char *path);

	void cmsending_(int *ecode)
	{
		edm::LogError("Generator|LHEInterface")
			<< "Herwig6 stopped run after recieving error code "
			<< *ecode << std::endl;
		throw cms::Exception("Herwig6Error")
			<< "Herwig stoped run with error code "
			<< *ecode << "." << std::endl;
	}
	void hwaend_() {}

	void upinit_() { fortranCallback.upinit(); }
	void upevnt_() { fortranCallback.upevnt(); }
} // extern "C"

#ifdef _POSIX_C_SOURCE
// some deep POSIX hackery to catch HERWIG sometimes (O(10k events) with
// complicated topologies) getting caught in and endless loop :-(
extern "C" {
	sigjmp_buf		_timeout_longjmp;

	static void _timeout_sighandler(int signr) {
		siglongjmp(_timeout_longjmp, 1);
	}

	static bool timeout(unsigned int secs, void (*callback)(void))
	{
		struct sigaction saOld = { 0, };

		struct itimerval itv;
		timerclear(&itv.it_value);
		timerclear(&itv.it_interval);
		itv.it_value.tv_sec = 0;
		itv.it_interval.tv_sec = 0;
		setitimer(ITIMER_VIRTUAL, &itv, NULL);

		sigset_t ss;
		sigemptyset(&ss);
		sigaddset(&ss, SIGVTALRM);

		sigprocmask(SIG_UNBLOCK, &ss, NULL);
		sigprocmask(SIG_BLOCK, &ss, NULL);

		if (sigsetjmp(_timeout_longjmp, 1)) {
			itv.it_value.tv_sec = 0;
			itv.it_interval.tv_sec = 0;
			setitimer(ITIMER_VIRTUAL, &itv, NULL);
			sigprocmask(SIG_UNBLOCK, &ss, NULL);
			return true;
		}

		itv.it_value.tv_sec = secs;
		itv.it_interval.tv_sec = secs;
		setitimer(ITIMER_VIRTUAL, &itv, NULL);

		struct sigaction sa = { 0, };
		sa.sa_handler = _timeout_sighandler;
		sa.sa_flags = SA_ONESHOT;
		sigemptyset(&sa.sa_mask);

		sigaction(SIGVTALRM, &sa, &saOld);
		sigprocmask(SIG_UNBLOCK, &ss, NULL);

		callback();

		itv.it_value.tv_sec = 0;
		itv.it_interval.tv_sec = 0;
		setitimer(ITIMER_VIRTUAL, &itv, NULL);

		sigaction(SIGVTALRM, &saOld, NULL);

		return false;
	}
} // extern "C"
#else
extern "C" {
	static bool timeout(unsigned int secs, void (*callback)(void))
	{
		callback();
		return false;
	}
} // extern "C"
#endif

static bool hwgive(const std::string &paramString);
static bool setRngSeeds(int mseed);

Herwig6Hadronisation::Herwig6Hadronisation(const edm::ParameterSet &params) :
	Hadronisation(params),
	herwigVerbosity(params.getUntrackedParameter<int>("herwigVerbosity", 0)),
	maxEventsToPrint(params.getUntrackedParameter<int>("maxEventsToPrint", 0)),
	useJimmy(params.getUntrackedParameter<bool>("useJimmy", true)),
	doMPInteraction(params.getUntrackedParameter<bool>("doMPInteraction", true)),
	numTrials(params.getUntrackedParameter<int>("numTrialsMPI", 100)),
	builtinPDFs(params.getUntrackedParameter<bool>("builtinPDFs", true)),
	lhapdfSetPath(params.getUntrackedParameter<std::string>("lhapdfSetPath", "")),
	needClear(false),
	buffer(0)
{
	std::vector<std::string> setNames =
		params.getParameter< std::vector<std::string> >("parameterSets");

	for(std::vector<std::string>::const_iterator iter = setNames.begin();
	    iter != setNames.end(); ++iter) {
		std::vector<std::string> lines =
			params.getParameter< std::vector<std::string> >(*iter);

		for(std::vector<std::string>::const_iterator
		    line = lines.begin(); line != lines.end(); ++line) {
			if (line->substr(0, 6) == "NRN(1)" ||
			    line->substr(0, 6) == "NRN(2)")
				throw cms::Exception("HerwigError")
					<< "Attempted to set random number"
					   " using Herwig command 'NRN(.)'."
					   " Please use the"
					   " RandomNumberGeneratorService."
					<< std::endl;

			paramLines.push_back(*line);
		}
	}
}

Herwig6Hadronisation::~Herwig6Hadronisation()
{
	clear();
	delete[] buffer;
}

void Herwig6Hadronisation::doInit()
{
	if (wantsShoweredEvent())
		buffer = new char[sizeof hepevt];

	// Call hwudat to set up HERWIG block data
	hwudat();

	// Setting basic parameters (doesn't matter, overridden by LHEF)
	hwproc.PBEAM1 = 7000.0;
	hwproc.PBEAM2 = 7000.0;
	std::memset(hwbmch.PART1, ' ', 8);
	std::memset(hwbmch.PART2, ' ', 8);
	hwbmch.PART1[0] = 'P';
	hwbmch.PART2[0] = 'P';

	if (useJimmy)
		jmparm.MSFLAG = 1;

	// initialize other common blocks ...
	hwigin();
	hwevnt.MAXER = 1000000;

	if (!builtinPDFs) {
		// set the LHAPDF grid directory and path
		if (lhapdfSetPath == "") {
			const char *lhaPdfs = getenv("LHAPATH");
			if (!lhaPdfs)
				throw cms::Exception("ConfigError")
					<< "LHAPATH not set." << std::endl;
			lhapdfSetPath = std::string(lhaPdfs);
		}
		lhapdfSetPath.resize(232, ' ');
		setherwpdf_();
		mysetpdfpath_(lhapdfSetPath.c_str());
	}

	if (useJimmy)
		jimmin();
	
	// set some 'non-herwig' defaults
	hwevnt.MAXPR = maxEventsToPrint;
	hwpram.IPRINT = herwigVerbosity;
	hwprop.RMASS[6] = 175.0;

	for(std::vector<std::string>::const_iterator iter = paramLines.begin();
	    iter != paramLines.end(); ++iter) {
		if (!hwgive(*iter))
			throw cms::Exception("HerwigError")
				<< "Herwig did not accept \""
				<< *iter << "\"." << std::endl;
	}

	edm::Service<edm::RandomNumberGenerator> rng;
	if (!setRngSeeds(rng->mySeed()))
		throw cms::Exception("HerwigError")
			<< "Impossible error in setting 'NRN(.)'.";
}

void Herwig6Hadronisation::clear()
{
	if (!needClear)
		return;

	// teminate elementary process
	hwefin();
	if (useJimmy)
		jmefin();

	needClear = false;
}

// naive Herwig6 HepMC status fixup
static int getStatus(const HepMC::GenParticle *p)
{
	int status = p->status();
	if (status == 1 || !p->end_vertex())
		return 1;
	else if (status == 3)
		return 3;
	else
		return 2;
}

static void fixupStatus(HepMC::GenEvent *event)
{
	for(HepMC::GenEvent::particle_iterator iter = event->particles_begin();
	    iter != event->particles_end(); iter++)
		(*iter)->set_status(getStatus(*iter));
}

static unsigned long hepevtSize()
{
	int n = HepMC::HEPEVT_Wrapper::max_number_entries();
	return sizeof(long) * (2 + 4 * n) + sizeof(double) * (9 * n);
}

std::auto_ptr<HepMC::GenEvent> Herwig6Hadronisation::doHadronisation()
{
	std::auto_ptr<HepMC::GenEvent> event;

	try {
		assert(!fortranCallback.instance);
		fortranCallback.instance = this;

		int counter = 0;
		while(counter++ < numTrials) {
			// call herwig routines to create HEPEVT
			hwuine();	// initialize event
			if (timeout(10, hwepro)) { // process event and PS
				// We hung for more than 10 seconds
				int error = 199;
				hwwarn_("HWHGUP", &error);
			}

			bool vetoed = false;	// matching and veto
			if (wantsShoweredEvent() && !hwevnt.IERROR) {
				// save HEPEVT since repair_hepevt breaks it
				std::memcpy(buffer, hepevt.data, hepevtSize());
				boost::shared_ptr<HepMC::GenEvent>
						event(conv.read_next_event());
				std::memcpy(hepevt.data, buffer, hepevtSize());
				fixupStatus(event.get());
				if (showeredEvent(event)) {
					hwevnt.IERROR = 198;
					vetoed = true;
				}
			}

			hwbgen();	// parton cascades

			// call jimmy ... only if event is not killed yet by HERWIG
			if (useJimmy && doMPInteraction && !hwevnt.IERROR &&
			    hwmsct_dummy(1.1) > 0.5)
				continue;

			hwdhob();	// heavy quark decays
			hwcfor();	// cluster formation
			hwcdec();	// cluster decays
			hwdhad();	// unstable particle decays
			hwdhvy();	// heavy flavour decays
			hwmevt();	// soft underlying event
			hwufne();	// finalize event

			if (vetoed) {
				fortranCallback.instance = 0;
				return event;
			}

			// if event was not killed by HERWIG retry
			if (!hwevnt.IERROR)
				break;
		}

		fortranCallback.instance = 0;

		if (counter >= numTrials) {
			edm::LogWarning("Generator|LHEInterface")
				<< "JIMMY could not produce MI in "
				<< numTrials << " trials." << std::endl
				<< "Event will be skipped to prevent"
				<< " from deadlock." << std::endl;

			return event;
		}
	} catch(...) {
		fortranCallback.instance = 0;
		throw;
	}

	event.reset(new HepMC::GenEvent);
	if (!conv.fill_next_event(event.get()))
		throw cms::Exception("HerwigError")
			<< "HepMC Conversion problems in event." << std::endl;

	fixupStatus(event.get());
	LHEEvent::fixHepMCEventTimeOrdering(event.get());

	getRawEvent()->fillEventInfo(event.get());

	HepMC::PdfInfo pdf;
	getRawEvent()->fillPdfInfo(&pdf);
	event->set_pdf_info(pdf);

	return event;
}

void Herwig6Hadronisation::newCommon(const boost::shared_ptr<LHECommon> &common)
{
	clear();
	try {
		assert(!fortranCallback.instance);
		fortranCallback.instance = this;

		needClear = true;

		// set user process to LHEF reader
		hwproc.IPROC = -1;

		hwuinc();
		hwusta("PI0     ", 8);
		hweini();
		if (useJimmy)
			jminit();

		fortranCallback.instance = 0;
	} catch(...) {
		fortranCallback.instance = 0;
		throw;
	}
}

static bool hwgive(const std::string &paramString)
{
	const char *param = paramString.c_str();

	if (!std::strncmp(param, "IPROC", 5)) {
		throw cms::Exception("HerwigError")
			<< "IPROC not supported since LHE file is read."
			<< std::endl;
	} else if (!std::strncmp(param, "TAUDEC", 6)) {
		int tostart = 0;
		while(param[tostart] != '=')
			tostart++;
		tostart++;
		while(param[tostart] == ' ')
			tostart++;
		int todo = 0;
		while(param[todo + tostart]) {
			hwdspn.TAUDEC[todo] = param[todo + tostart];
			todo++;
		}
		if (todo != 6)
			throw cms::Exception("HerwigError")
				<< "Attempted to set TAUDEC to "
				<< hwdspn.TAUDEC << ". This is not"
				   " allowed." << std::endl
				<< " Options for TAUDEC are HERWIG"
				   " and TAUOLA." <<std::endl;
	} else if (!std::strncmp(param, "BDECAY", 6))
		edm::LogWarning("Generator|LHEInterface")
			<< "BDECAY parameter *not* suported."
			   " HERWIG will use default b decay." << std::endl;
	else if (!std::strncmp(param, "QCDLAM", 6))
		hwpram.QCDLAM = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "VQCUT", 5))
		hwpram.VQCUT = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "VGCUT", 5))
		hwpram.VGCUT = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "VPCUT", 5))
		hwpram.VPCUT = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "CLMAX", 5))
		hwpram.CLMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "CLPOW", 5))
		hwpram.CLPOW = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PSPLT(1)", 8))
		hwpram.PSPLT[0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PSPLT(2)", 8))
		hwpram.PSPLT[1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "QDIQK", 5))
		hwpram.QDIQK = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PDIQK", 5))
		hwpram.PDIQK = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "QSPAC", 5))
		hwpram.QSPAC = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PTRMS", 5))
		hwpram.PTRMS = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IPRINT", 6))
		hwpram.IPRINT = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PRVTX", 5))
		hwpram.PRVTX = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NPRFMT", 6))
		hwpram.NPRFMT = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PRNDEC", 6))
		hwpram.PRNDEC = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PRNDEF", 6))
		hwpram.PRNDEF = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PRNTEX", 6))
		hwpram.PRNTEX = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PRNWEB", 6))
		hwpram.PRNWEB = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "MAXPR", 5))
		hwevnt.MAXPR = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "MAXER", 5))
		hwevnt.MAXER = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "LWEVT", 5))
		hwevnt.LWEVT = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "LRSUD", 5))
		hwpram.LRSUD = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "LWSUD", 5))
		hwpram.LWSUD = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NRN(1)", 6))
		hwevnt.NRN[0] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NRN(2)", 6))
		hwevnt.NRN[1] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "WGTMAX", 6))
		hwevnt.WGTMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NOWGT", 5))
		hwevnt.NOWGT = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "AVWGT", 5))
		hwevnt.AVWGT = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "AZSOFT", 6))
		hwpram.AZSOFT = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "AZSPIN", 6))
		hwpram.AZSPIN = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "HARDME", 6))
		hwpram.HARDME = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "SOFTME", 6))
		hwpram.SOFTME = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "GCUTME", 6))
		hwpram.GCUTME = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NCOLO", 5))
		hwpram.NCOLO = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NFLAV", 5))
		hwpram.NFLAV = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "MODPDF(1)", 9))
		hwpram.MODPDF[0] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "MODPDF(2)", 9))
		hwpram.MODPDF[1] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NSTRU", 5))
		hwpram.NSTRU = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PRSOF", 5))
		hwpram.PRSOF = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ENSOF", 5))
		hwpram.ENSOF = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IOPREM", 6))
		hwpram.IOPREM = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "BTCLM", 5))
		hwpram.BTCLM = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ETAMIX", 6))
		hwpram.ETAMIX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PHIMIX", 6))
		hwpram.PHIMIX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "H1MIX", 5))
		hwpram.H1MIX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "F0MIX", 5))
		hwpram.F0MIX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "F1MIX", 5))
		hwpram.F1MIX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "F2MIX", 5))
		hwpram.F2MIX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ET2MIX", 6))
		hwpram.ET2MIX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "OMHMIX", 6))
		hwpram.OMHMIX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PH3MIX", 6))
		hwpram.PH3MIX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "B1LIM", 5))
		hwpram.B1LIM = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "CLDIR(1)", 8))
		hwpram.CLDIR[0] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "CLDIR(2)", 8))
		hwpram.CLDIR[1] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "CLSMR(1)", 8))
		hwpram.CLSMR[0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "CLSMR(2)", 8))
		hwpram.CLSMR[1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(1)", 8))
		hwprop.RMASS[1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(2)", 8))
		hwprop.RMASS[2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(3)", 8))
		hwprop.RMASS[3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(4)", 8))
		hwprop.RMASS[4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(5)", 8))
		hwprop.RMASS[5] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(6)", 8))
		hwprop.RMASS[6] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(13)", 9))
		hwprop.RMASS[13] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "SUDORD", 6))
		hwusud.SUDORD = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "INTER", 5))
		hwusud.INTER = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NEGWTS", 6))
		hw6203.NEGWTS = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBN1", 5))
		hwminb.PMBN1 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBN2", 5))
		hwminb.PMBN2 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBN3", 5))
		hwminb.PMBN3 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBK1", 5))
		hwminb.PMBK1 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBK2", 5))
		hwminb.PMBK2 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBM1", 5))
		hwminb.PMBM1 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBM2", 5))
		hwminb.PMBM2 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBP1", 5))
		hwminb.PMBP1 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBP2", 5))
		hwminb.PMBP2 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PMBP3", 5))
		hwminb.PMBP3 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "VMIN2", 5))
		hwdist.VMIN2 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "EXAG", 4))
		hwdist.EXAG = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PRECO", 5))
		hwuclu.PRECO = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "CLRECO", 6))
		hwuclu.CLRECO = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PWT(1)", 6))
		hwuwts.PWT[0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PWT(2)", 6))
		hwuwts.PWT[1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PWT(3)", 6))
		hwuwts.PWT[2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PWT(4)", 6))
		hwuwts.PWT[3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PWT(5)", 6))
		hwuwts.PWT[4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PWT(6)", 6))
		hwuwts.PWT[5] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PWT(7)", 6))
		hwuwts.PWT[6] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,0,0)", 12))
		hwuwts.REPWT[0][0][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,0,1)", 12))
		hwuwts.REPWT[0][0][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,0,2)", 12))
		hwuwts.REPWT[0][0][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,0,3)", 12))
		hwuwts.REPWT[0][0][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,0,4)", 12))
		hwuwts.REPWT[0][0][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,1,0)", 12))
		hwuwts.REPWT[0][1][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,1,1)", 12))
		hwuwts.REPWT[0][1][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,1,2)", 12))
		hwuwts.REPWT[0][1][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,1,3)", 12))
		hwuwts.REPWT[0][1][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,1,4)", 12))
		hwuwts.REPWT[0][1][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,2,0)", 12))
		hwuwts.REPWT[0][2][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,2,1)", 12))
		hwuwts.REPWT[0][2][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,2,2)", 12))
		hwuwts.REPWT[0][2][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,2,3)", 12))
		hwuwts.REPWT[0][2][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,2,4)", 12))
		hwuwts.REPWT[0][2][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,3,0)", 12))
		hwuwts.REPWT[0][3][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,3,1)", 12))
		hwuwts.REPWT[0][3][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,3,2)", 12))
		hwuwts.REPWT[0][3][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,3,3)", 12))
		hwuwts.REPWT[0][3][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,3,4)", 12))
		hwuwts.REPWT[0][3][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,4,0)", 12))
		hwuwts.REPWT[0][4][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,4,1)", 12))
		hwuwts.REPWT[0][4][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,4,2)", 12))
		hwuwts.REPWT[0][4][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,4,3)", 12))
		hwuwts.REPWT[0][4][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(0,4,4)", 12))
		hwuwts.REPWT[0][4][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,0,0)", 12))
		hwuwts.REPWT[1][0][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,0,1)", 12))
		hwuwts.REPWT[1][0][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,0,2)", 12))
		hwuwts.REPWT[1][0][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,0,3)", 12))
		hwuwts.REPWT[1][0][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,0,4)", 12))
		hwuwts.REPWT[1][0][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,1,0)", 12))
		hwuwts.REPWT[1][1][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,1,1)", 12))
		hwuwts.REPWT[1][1][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,1,2)", 12))
		hwuwts.REPWT[1][1][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,1,3)", 12))
		hwuwts.REPWT[1][1][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,1,4)", 12))
		hwuwts.REPWT[1][1][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,2,0)", 12))
		hwuwts.REPWT[1][2][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,2,1)", 12))
		hwuwts.REPWT[1][2][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,2,2)", 12))
		hwuwts.REPWT[1][2][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,2,3)", 12))
		hwuwts.REPWT[1][2][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,2,4)", 12))
		hwuwts.REPWT[1][2][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,3,0)", 12))
		hwuwts.REPWT[1][3][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,3,1)", 12))
		hwuwts.REPWT[1][3][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,3,2)", 12))
		hwuwts.REPWT[1][3][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,3,3)", 12))
		hwuwts.REPWT[1][3][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,3,4)", 12))
		hwuwts.REPWT[1][3][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,4,0)", 12))
		hwuwts.REPWT[1][4][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,4,1)", 12))
		hwuwts.REPWT[1][4][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,4,2)", 12))
		hwuwts.REPWT[1][4][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,4,3)", 12))
		hwuwts.REPWT[1][4][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(1,4,4)", 12))
		hwuwts.REPWT[1][4][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,0,0)", 12))
		hwuwts.REPWT[2][0][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,0,1)", 12))
		hwuwts.REPWT[2][0][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,0,2)", 12))
		hwuwts.REPWT[2][0][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,0,3)", 12))
		hwuwts.REPWT[2][0][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,0,4)", 12))
		hwuwts.REPWT[2][0][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,1,0)", 12))
		hwuwts.REPWT[2][1][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,1,1)", 12))
		hwuwts.REPWT[2][1][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,1,2)", 12))
		hwuwts.REPWT[2][1][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,1,3)", 12))
		hwuwts.REPWT[2][1][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,1,4)", 12))
		hwuwts.REPWT[2][1][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,2,0)", 12))
		hwuwts.REPWT[2][2][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,2,1)", 12))
		hwuwts.REPWT[2][2][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,2,2)", 12))
		hwuwts.REPWT[2][2][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,2,3)", 12))
		hwuwts.REPWT[2][2][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,2,4)", 12))
		hwuwts.REPWT[2][2][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,3,0)", 12))
		hwuwts.REPWT[2][3][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,3,1)", 12))
		hwuwts.REPWT[2][3][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,3,2)", 12))
		hwuwts.REPWT[2][3][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,3,3)", 12))
		hwuwts.REPWT[2][3][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,3,4)", 12))
		hwuwts.REPWT[2][3][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,4,0)", 12))
		hwuwts.REPWT[2][4][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,4,1)", 12))
		hwuwts.REPWT[2][4][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,4,2)", 12))
		hwuwts.REPWT[2][4][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,4,3)", 12))
		hwuwts.REPWT[2][4][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(2,4,4)", 12))
		hwuwts.REPWT[2][4][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,0,0)", 12))
		hwuwts.REPWT[3][0][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,0,1)", 12))
		hwuwts.REPWT[3][0][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,0,2)", 12))
		hwuwts.REPWT[3][0][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,0,3)", 12))
		hwuwts.REPWT[3][0][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,0,4)", 12))
		hwuwts.REPWT[3][0][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,1,0)", 12))
		hwuwts.REPWT[3][1][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,1,1)", 12))
		hwuwts.REPWT[3][1][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,1,2)", 12))
		hwuwts.REPWT[3][1][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,1,3)", 12))
		hwuwts.REPWT[3][1][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,1,4)", 12))
		hwuwts.REPWT[3][1][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,2,0)", 12))
		hwuwts.REPWT[3][2][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,2,1)", 12))
		hwuwts.REPWT[3][2][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,2,2)", 12))
		hwuwts.REPWT[3][2][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,2,3)", 12))
		hwuwts.REPWT[3][2][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,2,4)", 12))
		hwuwts.REPWT[3][2][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,3,0)", 12))
		hwuwts.REPWT[3][3][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,3,1)", 12))
		hwuwts.REPWT[3][3][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,3,2)", 12))
		hwuwts.REPWT[3][3][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,3,3)", 12))
		hwuwts.REPWT[3][3][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,3,4)", 12))
		hwuwts.REPWT[3][3][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,4,0)", 12))
		hwuwts.REPWT[3][4][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,4,1)", 12))
		hwuwts.REPWT[3][4][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,4,2)", 12))
		hwuwts.REPWT[3][4][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,4,3)", 12))
		hwuwts.REPWT[3][4][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "REPWT(3,4,4)", 12))
		hwuwts.REPWT[3][4][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "SNGWT", 5))
		hwuwts.SNGWT = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "DECWT", 5))
		hwuwts.DECWT = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PLTCUT", 6))
		hwdist.PLTCUT = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "VTOCDK(", 7)){
		// we find the index ...
		int ind = std::atoi(&param[7]);
		hwprop.VTOCDK[ind] = std::atoi(&param[std::strcspn(param, "=") + 1]);} else if (!std::strncmp(param, "VTORDK(", 7)){
		// we find the index ...
		int ind = std::atoi(&param[7]);
		hwprop.VTORDK[ind] = std::atoi(&param[std::strcspn(param, "=") + 1]);} else if (!std::strncmp(param, "PIPSMR", 6))
		hwdist.PIPSMR = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "VIPWID(1)", 9))
		hw6202.VIPWID[0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "VIPWID(2)", 9))
		hw6202.VIPWID[1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "VIPWID(3)", 9))
		hw6202.VIPWID[2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "MAXDKL", 6))
		hwdist.MAXDKL = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IOPDKL", 6))
		hwdist.IOPDKL = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "DXRCYL", 6))
		hw6202.DXRCYL = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "DXZMAX", 6))
		hw6202.DXZMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "DXRSPH", 6))
		hw6202.DXRSPH = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "MIXING", 6))
		hwdist.MIXING = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "XMIX(1)", 7))
		hwdist.XMIX[0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "XMIX(2)", 7))
		hwdist.XMIX[1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "YMIX(1)", 7))
		hwdist.YMIX[0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "YMIX(2)", 7))
		hwdist.YMIX[1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(198)", 10))
		hwprop.RMASS[198] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(199)", 10))
		hwprop.RMASS[199] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "GAMW", 4))
		hwpram.GAMW = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "GAMZ", 4))
		hwpram.GAMZ = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(200)", 10))
		hwprop.RMASS[200] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "WZRFR", 5))
		hw6202.WZRFR = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "MODBOS(", 7)) {
		int ind = std::atoi(&param[7]);
		hwbosc.MODBOS[ind-1] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "RMASS(201)", 10))
		hwprop.RMASS[201] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IOPHIG", 6))
		hwbosc.IOPHIG = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "GAMMAX", 6))
		hwbosc.GAMMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ENHANC(", 7)) {
		int ind = std::atoi(&param[7]);
		hwbosc.ENHANC[ind-1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "RMASS(209)", 10))
		hwprop.RMASS[209] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(215)", 10))
		hwprop.RMASS[215] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ALPHEM", 6))
		hwpram.ALPHEM = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "SWEIN", 5))
		hwpram.SWEIN = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "QFCH(", 5)){
		int ind = std::atoi(&param[5]);
		hwpram.QFCH[ind-1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(1,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(2,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(3,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(4,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(5,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(6,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][5] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(7,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][6] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(8,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][7] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(9,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][8] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(10,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][9] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(11,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][10] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(12,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][11] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(13,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][12] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(14,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][13] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(15,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][14] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "AFCH(16,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.AFCH[ind-1][15] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(1,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][0] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(2,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(3,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][2] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(4,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][3] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(5,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][4] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(6,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][5] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(7,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][6] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(8,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][7] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(9,", 7)){
		int ind = std::atoi(&param[7]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][8] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(10,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][9] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(11,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][10] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(12,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][11] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(13,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][12] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(14,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][13] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(15,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][14] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "VFCH(16,", 8)){
		int ind = std::atoi(&param[8]);
		if (ind < 1 || ind > 2)
			return 0;
		hwpram.VFCH[ind-1][15] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "ZPRIME", 6))
		hwpram.ZPRIME = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RMASS(202)", 10))
		hwprop.RMASS[202] = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "GAMZP", 5))
		hwpram.GAMZP = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "VCKM(", 5)) {
		int ind1 = std::atoi(&param[5]);
		if (ind1 < 1 || ind1 > 3)
			return 0;
		int ind2 = std::atoi(&param[7]);
		if (ind2 < 1 || ind2 > 3)
			return 0;
		hwpram.VCKM[ind2][ind1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "SCABI", 5))
		hwpram.SCABI = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "EPOLN(", 6)) {
		int ind = std::atoi(&param[6]);
		if (ind < 1 || ind > 3)
			return 0;
		hwhard.EPOLN[ind-1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "PPOLN(", 6)) {
		int ind = std::atoi(&param[6]);
		if (ind < 1 || ind > 3)
			return 0;
		hwhard.PPOLN[ind-1] = std::atof(&param[std::strcspn(param, "=") + 1]);
	} else if (!std::strncmp(param, "QLIM", 4))
		hwhard.QLIM = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "THMAX", 5))
		hwhard.THMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "Y4JT", 4))
		hwhard.Y4JT = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "DURHAM", 6))
		hwhard.DURHAM = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IOP4JT(1)", 9))
		hwpram.IOP4JT[0] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IOP4JT(2)", 9))
		hwpram.IOP4JT[1] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "BGSHAT", 6))
		hwhard.BGSHAT = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "BREIT", 5))
		hwbrch.BREIT = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "USECMF", 6))
		hwbrch.USECMF = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NOSPAC", 6))
		hwpram.NOSPAC = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ISPAC", 5))
		hwpram.ISPAC = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "TMNISR", 6))
		hwhard.TMNISR = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ZMXISR", 6))
		hwhard.ZMXISR = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ASFIXD", 6))
		hwhard.ASFIXD = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "OMEGA0", 6))
		hwhard.OMEGA0 = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IAPHIG", 6))
		hwhard.IAPHIG = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PHOMAS", 6))
		hwhard.PHOMAS = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PRESPL", 6))
		hw6500.PRESPL = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PTMIN", 5))
		hwhard.PTMIN = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PTMAX", 5))
		hwhard.PTMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PTPOW", 5))
		hwhard.PTPOW = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "YJMIN", 5))
		hwhard.YJMIN = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "YJMAX", 5))
		hwhard.YJMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "EMMIN", 5))
		hwhard.EMMIN = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "EMMAX", 5))
		hwhard.EMMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "EMPOW", 5))
		hwhard.EMPOW = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "Q2MIN", 5))
		hwhard.Q2MIN = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "Q2MAX", 5))
		hwhard.Q2MAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "Q2POW", 5))
		hwhard.Q2POW = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "YBMIN", 5))
		hwhard.YBMIN = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "YBMAX", 5))
		hwhard.YBMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "WHMIN", 5))
		hwhard.WHMIN = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ZJMAX", 5))
		hwhard.ZJMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "Q2WWMN", 6))
		hwhard.Q2WWMN = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "Q2WWMX", 6))
		hwhard.Q2WWMX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "YWWMIN", 6))
		hwhard.YWWMIN = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "YWWMAX", 6))
		hwhard.YWWMAX = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "CSPEED", 6))
		hwpram.CSPEED = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "GEV2NB", 6))
		hwpram.GEV2NB = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IBSH", 4))
		hwhard.IBSH = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IBRN(1)", 7))
		hwhard.IBRN[0] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IBRN(2)", 7))
		hwhard.IBRN[1] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NQEV", 4))
		hwusud.NQEV = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ZBINM", 5))
		hwpram.ZBINM = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NZBIN", 5))
		hwpram.NZBIN = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NBTRY", 5))
		hwpram.NBTRY = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NCTRY", 5))
		hwpram.NCTRY = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NETRY", 5))
		hwpram.NETRY = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "NSTRY", 5))
		hwpram.NSTRY = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "ACCUR", 5))
		hwusud.ACCUR = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "RPARTY", 6))
		hwrpar.RPARTY = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "SUSYIN", 6))
		hwsusy.SUSYIN = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "LRSUSY", 6))
		hw6202.LRSUSY = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "SYSPIN", 6))
		hwdspn.SYSPIN = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "THREEB", 6))
		hwdspn.THREEB = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "FOURB", 5))
		hwdspn.FOURB = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "LHSOFT", 6))
		hwgupr.LHSOFT = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "LHGLSF", 6))
		hwgupr.LHGLSF = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "OPTM", 4))
		hw6300.OPTM = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IOPSTP", 6))
		hw6300.IOPSTP = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "IOPSH", 5))
		hw6300.IOPSH = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "JMUEO", 5))
		jmparm.JMUEO = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "PTJIM", 5))
		jmparm.PTJIM = std::atof(&param[std::strcspn(param, "=") + 1]);
	else if (!std::strncmp(param, "JMRAD(73)", 9))
		jmparm.JMRAD[72] = std::atoi(&param[std::strcspn(param, "=") + 1]);
	else
		return false;

	return true;
}

static bool setRngSeeds(int mseed)
{
	double temx[5];
	for(int i = 0; i < 5; i++) {
		mseed = mseed * 29943829 - 1;
		temx[i] = mseed * (1.0 / (65536.0 * 65536.0));
	}
	long double c;
	c = (long double)2111111111.0 * temx[3] +
	    1492.0 * (temx[3] = temx[2]) +
	    1776.0 * (temx[2] = temx[1]) +
	    5115.0 * (temx[1] = temx[0]) +
	    temx[4];
	temx[4] = std::floor(c);
	temx[0] = c - temx[4];
	temx[4] = temx[4] * (1.0 / (65536.0 * 65536.0));
	hwevnt.NRN[0] = (int)(temx[0] * 99999);
	c = (long double)2111111111.0 * temx[3] +
	    1492.0 * (temx[3] = temx[2]) +
	    1776.0 * (temx[2] = temx[1]) +
	    5115.0 * (temx[1] = temx[0]) +
	    temx[4];
	temx[4] = std::floor(c);
	temx[0] = c - temx[4];
	hwevnt.NRN[1] = (int)(temx[0] * 99999);

	return true;
}

void Herwig6Hadronisation::fillHeader()
{
	const HEPRUP *heprup = getRawEvent()->getHEPRUP();

	CommonBlocks::fillHEPRUP(heprup);

	if (builtinPDFs) {
		heprup_.pdfgup[0] = -1;
		heprup_.pdfgup[1] = -1;
		heprup_.pdfsup[0] = -1;
		heprup_.pdfsup[1] = -1;
	}
}

void Herwig6Hadronisation::fillEvent()
{
	const HEPEUP *hepeup = getRawEvent()->getHEPEUP();

	CommonBlocks::fillHEPEUP(hepeup);
}

DEFINE_LHE_HADRONISATION_PLUGIN(Herwig6Hadronisation);

} // namespace lhef

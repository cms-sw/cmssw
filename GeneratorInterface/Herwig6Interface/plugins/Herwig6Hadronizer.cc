#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>
#include <HepMC/GenVertex.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/HerwigWrapper6_4.h>
#include <HepMC/HEPEVT_Wrapper.h>
#include <HepMC/IO_HERWIG.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Instance.h"
#include "GeneratorInterface/Herwig6Interface/interface/herwig.h"

extern "C" {
#ifdef HERWIG_DOES_NOT_CALL_PDFSET
	void pdfset_(char parm[20][20], double value[20]);
#endif
	void hwuidt_(int *iopt, int *ipdg, int *iwig, char nwig[8]);
}

// helpers
namespace {
	static inline bool call_hwmsct()
	{
		int result;
		hwmsct(&result);
		return result;
	}

	static int pdgToHerwig(int ipdg, char *nwig)
	{
		int iopt = 1;
		int iwig = 0;
		hwuidt_(&iopt, &ipdg, &iwig, nwig);
		return ipdg ? iwig : 0;
	}

	static bool markStable(int pdgId)
	{
		char nwig[9] = "        ";
		if (!pdgToHerwig(pdgId, nwig))
			return false;
		hwusta(nwig, 1);
		return true;
	}
}

class Herwig6Hadronizer : public gen::BaseHadronizer,
                          public gen::Herwig6Instance {
    public:
	Herwig6Hadronizer(const edm::ParameterSet &params);
	~Herwig6Hadronizer();

	bool initializeForInternalPartons();
//	bool initializeForExternalPartons();
	bool declareStableParticles(const std::vector<int> &pdgIds);

	void statistics();

	bool generatePartonsAndHadronize();
//	bool hadronize();
	bool decay();
	bool residualDecay();
	void finalizeEvent();

	const char *classname() const { return "Herwig6Hadronizer"; }

    private:
	void clear();

	int pythiaStatusCode(const HepMC::GenParticle *p) const;
	void pythiaStatusCodes();

//	virtual void upInit();
//	virtual void upEvnt();

	HepMC::IO_HERWIG		conv;
	bool				needClear;
	bool				externalPartons;

	gen::ParameterCollector		parameters;
	int				herwigVerbosity;
	int				hepmcVerbosity;
	int				maxEventsToPrint;
	bool				printCards;
	bool				emulatePythiaStatusCodes;
	double				comEnergy;
	bool				useJimmy;
	bool				doMPInteraction;
	int				numTrials;
};

extern "C" {
	void hwwarn_(const char *method, int *id);
	void setherwpdf_(void);
	void mysetpdfpath_(const char *path);
} // extern "C"

Herwig6Hadronizer::Herwig6Hadronizer(const edm::ParameterSet &params) :
	needClear(false),
	parameters(params.getParameter<edm::ParameterSet>("HerwigParameters")),
	herwigVerbosity(params.getUntrackedParameter<int>("herwigVerbosity", 0)),
	hepmcVerbosity(params.getUntrackedParameter<int>("hepmcVerbosity", 0)),
	maxEventsToPrint(params.getUntrackedParameter<int>("maxEventsToPrint", 0)),
	printCards(params.getUntrackedParameter<bool>("printCards", false)),
	emulatePythiaStatusCodes(params.getUntrackedParameter<bool>("emulatePythiaStatusCodes", false)),
	comEnergy(params.getParameter<double>("comEnergy")),
	useJimmy(params.getParameter<bool>("useJimmy")),
	doMPInteraction(params.getParameter<bool>("doMPInteraction")),
	numTrials(params.getUntrackedParameter<int>("numTrialsMPI", 100))
{
	runInfo().setExternalXSecLO(
		params.getUntrackedParameter<double>("crossSection", -1.0));
	runInfo().setFilterEfficiency(
		params.getUntrackedParameter<double>("filterEfficiency", -1.0));
}

Herwig6Hadronizer::~Herwig6Hadronizer()
{
	clear();
}

void Herwig6Hadronizer::clear()
{
	if (!needClear)
		return;

	// teminate elementary process
	call(hwefin);
	if (useJimmy)
		call(jmefin);

	needClear = false;
}

bool Herwig6Hadronizer::initializeForInternalPartons()
{
	clear();

	externalPartons = false;

	std::ostringstream info;
	info << "---------------------------------------------------\n"; 
	info << "Initializing Herwig6Hadronizer for internal partons\n";
	info << "---------------------------------------------------\n";

	info << "   Herwig verbosity level         = " << herwigVerbosity << "\n";
	info << "   HepMC verbosity                = " << hepmcVerbosity << "\n";
	info << "   Number of events to be printed = " << maxEventsToPrint << "\n";

	// Call hwudat to set up HERWIG block data
	hwudat();

	// Setting basic parameters
	info << "   Center-of-Mass pp energy       = " << comEnergy << " GeV\n";
	hwproc.PBEAM2 = hwproc.PBEAM1 = 0.5 * comEnergy;
	std::memcpy(hwbmch.PART1, "P       ", 8);
	std::memcpy(hwbmch.PART2, "P       ", 8);

	if (useJimmy) {
		info << "   HERWIG will be using JIMMY for UE/MI.\n";
		jmparm.MSFLAG = 1;
		if (doMPInteraction)
			info << "   JIMMY trying to generate multiple interactions.\n";
	}

	// initialize other common blocks ...
	call(hwigin);
	hwevnt.MAXER = 100000000;	// O(inf)
	hwpram.LWSUD = 0;		// don't write Sudakov form factors
	hwdspn.LWDEC = 0;		// don't write three/four body decays
					// (no fort.77 and fort.88 ...)

	// init LHAPDF glue

#if HERWIG_DOES_NOT_CALL_PDFSET
	char parm[20][20];
	double value[20];
	std::memset(parm, ' ', sizeof parm);
	std::memset(value, 0, sizeof value);
	std::memcpy(parm[0], "HWLHAPDF", 8);
	pdfset_(parm, value);
#endif

	std::memset(hwprch.AUTPDF, ' ', sizeof hwprch.AUTPDF);
	for(unsigned int i = 0; i < 2; i++)
		std::memcpy(hwprch.AUTPDF[i], "HWLHAPDF", 8);

	if (useJimmy)
		call(jimmin);

	hwevnt.MAXPR = maxEventsToPrint;
	hwpram.IPRINT = herwigVerbosity;
//	hwprop.RMASS[6] = 175.0;

	if (printCards) {
		info << "\n";
		info << "------------------------------------\n";
		info << "Reading HERWIG parameters\n";
		info << "------------------------------------\n";
    
	}
	for(gen::ParameterCollector::const_iterator line = parameters.begin();
	    line != parameters.end(); ++line) {
		if (printCards)
			info << "   " << *line << "\n";
		if (!give(*line))
			throw edm::Exception(edm::errors::Configuration)
				<< "Herwig 6 did not accept the following: \""
				<< *line << "\"." << std::endl;
	}

	if (printCards)
		info << "\n";

	needClear = true;

	// HERWIG preparations ...
	call(hwuinc);
	markStable(111);	//FIXME?

	// initialize HERWIG event generation
	call(hweini);

	if (useJimmy)
		call(jminit);

	edm::LogInfo(info.str());

	return true;
}

bool Herwig6Hadronizer::declareStableParticles(const std::vector<int> &pdgIds)
{
	for(std::vector<int>::const_iterator iter = pdgIds.begin();
	    iter != pdgIds.end(); ++iter)
		if (!markStable(*iter))
			return false;
	return true;
}

void Herwig6Hadronizer::statistics()
{
	double RNWGT = 1. / hwevnt.NWGTS;
	double AVWGT = hwevnt.WGTSUM * RNWGT;

	double xsec = 1.0e3 * AVWGT;

	runInfo().setInternalXSec(xsec);
}

bool Herwig6Hadronizer::generatePartonsAndHadronize()
{
	// hard process generation, parton shower, hadron formation

	InstanceWrapper wrapper(this);	// safe guard

	event().reset();

	int counter = 0;
	while(counter++ < numTrials) {
		// call herwig routines to create HEPEVT

		hwuine();	// initialize event

		if (callWithTimeout(10, hwepro)) { // process event and PS
			// We hung for more than 10 seconds
			int error = 199;
			hwwarn_("HWHGUP", &error);
		}

		hwbgen();	// parton cascades

		// call jimmy ... only if event is not killed yet by HERWIG
		if (useJimmy && doMPInteraction && !hwevnt.IERROR &&
		    call_hwmsct())
				continue;

		hwdhob();	// heavy quark decays
		hwcfor();	// cluster formation
		hwcdec();	// cluster decays

		// if event was not killed by HERWIG, break out of retry loop
		if (!hwevnt.IERROR)
			break;

		hwufne();	// finalize event
	}

	if (counter >= numTrials) {
		edm::LogWarning("Generator|LHEInterface")
			<< "JIMMY could not produce MI in "
			<< numTrials << " trials." << std::endl
			<< "Event will be skipped to prevent"
			<< " from deadlock." << std::endl;

		return false;
	}

	return true;
}

void Herwig6Hadronizer::finalizeEvent()
{
	lhef::LHEEvent::fixHepMCEventTimeOrdering(event().get());

	if (emulatePythiaStatusCodes)
		pythiaStatusCodes();

	event()->set_signal_process_id(hwproc.IPROC);
}

bool Herwig6Hadronizer::decay()
{
	// hadron decays

	InstanceWrapper wrapper(this);	// safe guard

	hwdhad();	// unstable particle decays
	hwdhvy();	// heavy flavour decays
	hwmevt();	// soft underlying event

	hwufne();	// finalize event

	if (hwevnt.IERROR)
		return false;

	event().reset(new HepMC::GenEvent);
	if (!conv.fill_next_event(event().get()))
		throw cms::Exception("Herwig6Error")
			<< "HepMC Conversion problems in event." << std::endl;

	return true;
}

bool Herwig6Hadronizer::residualDecay()
{
	return true;
}

int Herwig6Hadronizer::pythiaStatusCode(const HepMC::GenParticle *p) const
{
	int status = p->status();

	// weird 9922212 particles...
	if (status == 3 && !p->end_vertex())
		status = 2;

	if (status >= 1 && status <= 3)
		return status;

	if (!p->end_vertex())
		return 1;

	if (externalPartons)
		return 2;

	for(;;) {
		if (p->pdg_id() == 0)
			break;

		if (status >= 120 && status <= 122 || status == 3)
			return 3;

		if (!(status == 123 || status == 124 ||
		      status == 155 || status == 156 || status == 160 ||
		      status >= 195 && status <= 197))
			break;

		const HepMC::GenVertex *vtx = p->production_vertex();
		if (!vtx || !vtx->particles_in_size())
			break;

		p = *vtx->particles_in_const_begin();
		status = p->status();
	}

	return 2; 
}

void Herwig6Hadronizer::pythiaStatusCodes()
{
	for(HepMC::GenEvent::particle_iterator iter =
	    					event()->particles_begin();
	    iter != event()->particles_end(); iter++)
		(*iter)->set_status(pythiaStatusCode(*iter));
}

typedef edm::GeneratorFilter<Herwig6Hadronizer> Herwig6GeneratorFilter;
DEFINE_FWK_MODULE(Herwig6GeneratorFilter);

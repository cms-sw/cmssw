#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>

#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

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

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Instance.h"
#include "GeneratorInterface/Herwig6Interface/interface/herwig.h"

extern "C" {
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

	void setSLHAFromHeader(const std::vector<std::string> &lines);
	bool initialize(const lhef::HEPRUP *heprup);

	bool initializeForInternalPartons() { return initialize(0); }
	bool initializeForExternalPartons() { return initialize(lheRunInfo()->getHEPRUP()); }

	bool declareStableParticles(const std::vector<int> &pdgIds);

	void statistics();


	bool generatePartonsAndHadronize() { return hadronize(); }
	bool hadronize();
	bool decay();
	bool residualDecay();
	void finalizeEvent();

	const char *classname() const { return "Herwig6Hadronizer"; }

    private:
	void clear();

	int pythiaStatusCode(const HepMC::GenParticle *p) const;
	void pythiaStatusCodes();

	virtual void upInit();
	virtual void upEvnt();

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

        bool                            readMCatNLOfile;

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
	numTrials(params.getUntrackedParameter<int>("numTrialsMPI", 100)),
	readMCatNLOfile(false)
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

void Herwig6Hadronizer::setSLHAFromHeader(
				const std::vector<std::string> &lines)
{
	std::set<std::string> blocks;
	std::string block;
	for(std::vector<std::string>::const_iterator iter = lines.begin();
	    iter != lines.end(); ++iter) {
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
			if (tokens[0] == "BLOCK")
				block = tokens[1];
			else if (tokens[0] == "DECAY")
				block = "DECAY";

			if (block.empty())
				continue;

			if (!blocks.count(block)) {
				blocks.insert(block);
				edm::LogWarning("Generator|Herwig6Hadronzier")
					<< "Unsupported SLHA block \"" << block
					<< "\".  It will be ignored."
					<< std::endl;
			}
		}
	}
}

bool Herwig6Hadronizer::initialize(const lhef::HEPRUP *heprup)
{
	clear();

	externalPartons = (heprup != 0);

	std::ostringstream info;
	info << "---------------------------------------------------\n"; 
	info << "Initializing Herwig6Hadronizer for "
	     << (externalPartons ? "external" : "internal") << " partons\n";
	info << "---------------------------------------------------\n";

	info << "   Herwig verbosity level         = " << herwigVerbosity << "\n";
	info << "   HepMC verbosity                = " << hepmcVerbosity << "\n";
	info << "   Number of events to be printed = " << maxEventsToPrint << "\n";

	// Call hwudat to set up HERWIG block data
	hwudat();

	// Setting basic parameters
	if (externalPartons) {
		hwproc.PBEAM1 = heprup->EBMUP.first;
		hwproc.PBEAM2 = heprup->EBMUP.second;
		pdgToHerwig(heprup->IDBMUP.first, hwbmch.PART1);
		pdgToHerwig(heprup->IDBMUP.second, hwbmch.PART2);
	} else {
		hwproc.PBEAM1 = 0.5 * comEnergy;
		hwproc.PBEAM2 = 0.5 * comEnergy;
		pdgToHerwig(2212, hwbmch.PART1);
		pdgToHerwig(2212, hwbmch.PART2);
	}

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

	std::memset(hwprch.AUTPDF, ' ', sizeof hwprch.AUTPDF);
	for(unsigned int i = 0; i < 2; i++) {
		hwpram.MODPDF[i] = -111;
		std::memcpy(hwprch.AUTPDF[i], "HWLHAPDF", 8);
	}

	if (useJimmy)
		call(jimmin);

	hwevnt.MAXPR = maxEventsToPrint;
	hwpram.IPRINT = herwigVerbosity;
//	hwprop.RMASS[6] = 175.0;	//FIXME

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

	if (externalPartons) {
		std::vector<std::string> slha =
				lheRunInfo()->findHeader("slha");
		if (!slha.empty())
			setSLHAFromHeader(slha);
	}

	needClear = true;

	std::pair<int, int> pdfs(-1, -1);
	if (externalPartons)
		pdfs = lheRunInfo()->pdfSetTranslation();

	if (hwpram.MODPDF[0] != -111 || hwpram.MODPDF[1] != -111) {
		for(unsigned int i = 0; i < 2; i++)
			if (hwpram.MODPDF[i] == -111)
				hwpram.MODPDF[i] = -1;

		if (pdfs.first != -1 || pdfs.second != -1)
			edm::LogError("Generator|Herwig6Hadronzier")
				<< "Both external Les Houches event and "
			           "config file specify a PDF set.  "
				   "User PDF will override external one."
				<< std::endl;

		pdfs.first = hwpram.MODPDF[0] != -111 ? hwpram.MODPDF[0] : -1;
		pdfs.second = hwpram.MODPDF[1] != -111 ? hwpram.MODPDF[1] : -1;
	}

	hwpram.MODPDF[0] = pdfs.first;
	hwpram.MODPDF[1] = pdfs.second;

 	if (externalPartons)
		hwproc.IPROC = -1;

	// HERWIG preparations ...
	call(hwuinc);
	markStable(111);	//FIXME?	only pi0?
	// better: merge with declareStableParticles
	// and get the list from configuration / Geant4 / Core somewhere

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

bool Herwig6Hadronizer::hadronize()
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
		edm::LogWarning("Generator|Herwig6Hadronizer")
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

void Herwig6Hadronizer::upInit()
{
	lhef::CommonBlocks::fillHEPRUP(lheRunInfo()->getHEPRUP());
	heprup_.pdfgup[0] = heprup_.pdfgup[1] = -1;
	heprup_.pdfsup[0] = heprup_.pdfsup[1] = -1;
	// we set up the PDFs ourselves
	
	// pass HERWIG paramaters fomr header (if present)
	std::string mcnloHeader="herwig6header";
	std::vector<lhef::LHERunInfo::Header> headers=lheRunInfo()->getHeaders();
	for(std::vector<lhef::LHERunInfo::Header>::const_iterator hIter=headers.begin();hIter!=headers.end(); ++hIter) {
	  if(hIter->tag()==mcnloHeader){
	    readMCatNLOfile=true;
	    for(lhef::LHERunInfo::Header::const_iterator lIter=hIter->begin(); lIter != hIter->end(); ++lIter) {
	      if((lIter->c_str())[1]!='#') {   // it's not a comment
		if (!give(*lIter))
		  throw edm::Exception(edm::errors::Configuration)
		    << "Herwig 6 did not accept the following: \""
		    << *lIter << "\"." << std::endl;
	      }
	    }
	  }
	}
}

void Herwig6Hadronizer::upEvnt()
{
	lhef::CommonBlocks::fillHEPEUP(lheEvent()->getHEPEUP());

	// if MCatNLO external file is read, read comment & pass IHPRO to HERWIG
	if(readMCatNLOfile) {
	  for(std::vector<std::string>::const_iterator iter=lheEvent()->getComments().begin();
	      iter!=lheEvent()->getComments().end(); ++iter) {
	    std::string toParse(iter->substr(1));
	    if (!give(toParse))
	      throw edm::Exception(edm::errors::Configuration)
		    << "Herwig 6 did not accept the following: \""
		    << toParse << "\"." << std::endl;
	  }
	}
	
}

int Herwig6Hadronizer::pythiaStatusCode(const HepMC::GenParticle *p) const
{
	int status = p->status();

	// weird 9922212 particles...
	if (status == 3 && !p->end_vertex())
		return 2;

	if (status >= 1 && status <= 3)
		return status;

	if (!p->end_vertex())
		return 1;

	// let's prevent particles having status 3, if the identical
	// particle downstream is a better status 3 candidate
	int currentId = p->pdg_id();
	int orig = status;
	if (status == 123 || status == 124 ||
	    status == 155 || status == 156 || status == 160 ||
	    (status >= 195 && status <= 197)) {
		for(const HepMC::GenParticle *q = p;;) {
			const HepMC::GenVertex *vtx = q->end_vertex();
			if (!vtx)
				break;

			HepMC::GenVertex::particles_out_const_iterator iter;
			for(iter = vtx->particles_out_const_begin();
			    iter != vtx->particles_out_const_end(); ++iter)
				if ((*iter)->pdg_id() == currentId)
					break;

			if (iter == vtx->particles_out_const_end())
				break;

			q = *iter;
			if (q->status() == 3 ||
			    (status == 120 || status == 123 ||
			     status == 124) && orig > 124)
				return 4;
		}
	}

	int nesting = 0;
	for(;;) {
		if (status >= 120 && status <= 122 || status == 3) {
			// avoid flagging status 3 if there is a
			// better status 3 candidate upstream
			if (externalPartons)
				return (orig >= 121 && orig <= 124 ||
				        orig == 3) ? 3 : 4;
			else
				return (nesting ||
				        status != 3 && orig <= 124) ? 3 : 4;
		}

		// check whether we are leaving the hard process
		// including heavy resonance decays
		if (!(status == 4 || status == 123 || status == 124 ||
		      status == 155 || status == 156 || status == 160 ||
		      (status >= 195 && status <= 197)))
			break;

		const HepMC::GenVertex *vtx = p->production_vertex();
		if (!vtx || !vtx->particles_in_size())
			break;

		p = *vtx->particles_in_const_begin();
		status = p->status();

		int newId = p->pdg_id();

		if (!newId)
			break;

		// nesting increases if we move to the next-best mother
		if (newId != currentId) {
			if (++nesting > 1 && externalPartons)
				break;
			currentId = newId;
		}
	}

	return 2; 
}

void Herwig6Hadronizer::pythiaStatusCodes()
{
	for(HepMC::GenEvent::particle_iterator iter =
	    					event()->particles_begin();
	    iter != event()->particles_end(); iter++)
		(*iter)->set_status(pythiaStatusCode(*iter));

	for(HepMC::GenEvent::particle_iterator iter =
	    					event()->particles_begin();
	    iter != event()->particles_end(); iter++)
		if ((*iter)->status() == 4)
			(*iter)->set_status(2);
}

typedef edm::GeneratorFilter<Herwig6Hadronizer> Herwig6GeneratorFilter;
DEFINE_FWK_MODULE(Herwig6GeneratorFilter);

typedef edm::HadronizerFilter<Herwig6Hadronizer> Herwig6HadronizerFilter;
DEFINE_FWK_MODULE(Herwig6HadronizerFilter);

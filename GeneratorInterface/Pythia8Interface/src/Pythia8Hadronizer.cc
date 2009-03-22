#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <stdint.h>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>

#include <Pythia.h>
#include <HepMCInterface.h>

#include <LesHouches.h>

#include "GeneratorInterface/Pythia8Interface/interface/RandomP8.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"
#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

using namespace gen;
using namespace Pythia8;

class Pythia8Hadronizer : public BaseHadronizer {
    public:
	Pythia8Hadronizer(const edm::ParameterSet &params);
	~Pythia8Hadronizer();
 
	bool initializeForInternalPartons();
    bool initializeForExternalPartons();
	bool declareStableParticles(const std::vector<int> &pdgIds);

	void statistics();

	bool generatePartonsAndHadronize();
    bool hadronize();
	bool decay();
	bool residualDecay();
	void finalizeEvent();

	const char *classname() const { return "Pythia8Hadronizer"; }

    private:
	ParameterCollector	parameters;

	/// Center-of-Mass energy
	double			comEnergy;
	/// Pythia PYLIST Verbosity flag
	unsigned int		pythiaPylistVerbosity;
	/// HepMC verbosity flag
	bool			pythiaHepMCVerbosity;
	/// Events to print if verbosity
	unsigned int		maxEventsToPrint;

    class LHAupLesHouches;

	std::auto_ptr<Pythia>	pythia;
	Event			*pythiaEvent;
	HepMC::I_Pythia8	toHepMC;   

    std::auto_ptr<LHAupLesHouches>      lhaUP;

};

class Pythia8Hadronizer::LHAupLesHouches : public LHAup {
    public:
    LHAupLesHouches() {}

    void loadRunInfo(const boost::shared_ptr<lhef::LHERunInfo> &runInfo)
    { this->runInfo = runInfo; }

    //void loadEvent(const boost::shared_ptr<LHEEvent> &event)
    //{ this->event = event; }

    private:

    bool setInit();
    bool setEvent(int idProcIn);

    //Hadronisation           *hadronisation;
    boost::shared_ptr<lhef::LHERunInfo>   runInfo;
    //boost::shared_ptr<LHEEvent> event;
};

bool Pythia8Hadronizer::LHAupLesHouches::setInit()
{
    //if (!runInfo)
    //    return false;
    //runInfo.reset();
    return true;
}

bool Pythia8Hadronizer::LHAupLesHouches::setEvent(int inProcId)
{
    //if (!event)
    //    return false;
    //event.reset();
    return true;
}


Pythia8Hadronizer::Pythia8Hadronizer(const edm::ParameterSet &params) :
	parameters(params.getParameter<edm::ParameterSet>("PythiaParameters")),
	comEnergy(params.getParameter<double>("comEnergy")),
	pythiaPylistVerbosity(params.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
	pythiaHepMCVerbosity(params.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false)),
	maxEventsToPrint(params.getUntrackedParameter<int>("maxEventsToPrint", 0))
{
    randomEngine = &getEngineReference();

	runInfo().setExternalXSecLO(
		params.getUntrackedParameter<double>("crossSection", -1.0));
	runInfo().setFilterEfficiency(
		params.getUntrackedParameter<double>("filterEfficiency", -1.0));
}

Pythia8Hadronizer::~Pythia8Hadronizer()
{
}

bool Pythia8Hadronizer::initializeForInternalPartons()
{
	//Old code that used Pythia8 own random engine
	//edm::Service<edm::RandomNumberGenerator> rng;
	//uint32_t seed = rng->mySeed();
	//Pythia8::Rndm::init(seed);

    RandomP8* RP8 = new RandomP8();

	pythia.reset(new Pythia);
    pythia->setRndmEnginePtr(RP8);
	pythiaEvent = &pythia->event;

	for(ParameterCollector::const_iterator line = parameters.begin();
	    line != parameters.end(); ++line) {
		if (line->find("Random:") != std::string::npos)
			throw cms::Exception("PythiaError")
				<< "Attempted to set random number "
				   "using Pythia commands.  Please use "
				   "the RandomNumberGeneratorService."
				<< std::endl;

		if (!pythia->readString(*line))
			throw cms::Exception("PythiaError")
				<< "Pythia 8 did not accept \""
				<< *line << "\"." << std::endl;
	}

	pythia->init(2212, 2212, comEnergy);

	pythia->settings.listChanged();

	return true;
}


bool Pythia8Hadronizer::initializeForExternalPartons()
{

    std::cout << "Initializing for external partons" << std::endl;

    RandomP8* RP8 = new RandomP8();

    pythia.reset(new Pythia);
    pythia->setRndmEnginePtr(RP8);
    pythiaEvent = &pythia->event;

    lhaUP.reset(new LHAupLesHouches());

    pythia->init("ttbar.lhe");

    return true;
}


#if 0
// naive Pythia8 HepMC status fixup
static int getStatus(const HepMC::GenParticle *p)
{
	int status = p->status();
	if (status > 0)
		return status;
	else if (status > -30 && status < 0)
		return 3;
	else
		return 2;
}
#endif

bool Pythia8Hadronizer::declareStableParticles(const std::vector<int> &pdgIds)
{
#if 0
	for(std::vector<int>::const_iterator iter = pdgIds.begin();
	    iter != pdgIds.end(); ++iter)
		if (!markStable(*iter))
			return false;

	return true;
#else
	return false;
#endif
}

void Pythia8Hadronizer::statistics()
{
	pythia->statistics();

	double xsec = pythia->info.sigmaGen(); // cross section in mb
	runInfo().setInternalXSec(xsec);
}

bool Pythia8Hadronizer::generatePartonsAndHadronize()
{
	if (!pythia->next())
		return false;

	event().reset(new HepMC::GenEvent);
	toHepMC.fill_next_event(*pythiaEvent, event().get());

	return true;
}

bool Pythia8Hadronizer::hadronize()
{
    if (!pythia->next())
        return false;

    event().reset(new HepMC::GenEvent);
    toHepMC.fill_next_event(*pythiaEvent, event().get());

    return true;
}

bool Pythia8Hadronizer::decay()
{
	return true;
}

bool Pythia8Hadronizer::residualDecay()
{
	return true;
}

void Pythia8Hadronizer::finalizeEvent()
{
#if 0
	for(HepMC::GenEvent::particle_iterator iter = event->particles_begin();
	    iter != event->particles_end(); iter++)
		(*iter)->set_status(getStatus(*iter));
#endif

	event()->set_signal_process_id(pythia->info.code());
	event()->set_event_scale(pythia->info.pTHat());	//FIXME

	int id1 = pythia->info.id1();
	int id2 = pythia->info.id2();
	if (id1 == 21) id1 = 0;
	if (id2 == 21) id2 = 0;
	double x1 = pythia->info.x1();
	double x2 = pythia->info.x2();
	double Q = pythia->info.QRen();
	double pdf1 = pythia->info.pdf1() / pythia->info.x1();
	double pdf2 = pythia->info.pdf2() / pythia->info.x2();
	event()->set_pdf_info(HepMC::PdfInfo(id1,id2,x1,x2,Q,pdf1,pdf2));

	event()->weights().push_back(pythia->info.weight());

	//******** Verbosity ********

	if (maxEventsToPrint > 0 &&
	    (pythiaPylistVerbosity || pythiaHepMCVerbosity)) {
		maxEventsToPrint--;
		if (pythiaPylistVerbosity) {
			pythia->info.list(std::cout); 
			pythia->event.list(std::cout);
		} 

		if (pythiaHepMCVerbosity) {
			std::cout << "Event process = "
			          << pythia->info.code() << "\n"
			          << "----------------------" << std::endl;
			event()->print();
		}
	}
}

typedef edm::GeneratorFilter<Pythia8Hadronizer> Pythia8GeneratorFilter;
DEFINE_FWK_MODULE(Pythia8GeneratorFilter);

typedef edm::HadronizerFilter<Pythia8Hadronizer> Pythia8HadronizerFilter;
DEFINE_FWK_MODULE(Pythia8HadronizerFilter);

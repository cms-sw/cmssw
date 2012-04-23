#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <stdint.h>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>

#include <Pythia.h>
#include <HepMCInterface.h>

#include "GeneratorInterface/Pythia8Interface/interface/RandomP8.h"

#include "GeneratorInterface/Pythia8Interface/interface/UserHooks.h"

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

#include "GeneratorInterface/Pythia8Interface/interface/LHAupLesHouches.h"

#include "HepPID/ParticleIDTranslations.hh"

#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

using namespace gen;
using namespace Pythia8;

class Pythia8Hadronizer : public BaseHadronizer {
    public:
	Pythia8Hadronizer(const edm::ParameterSet &params);
	~Pythia8Hadronizer();
 
	bool initializeForInternalPartons();
        bool initializeForExternalPartons();
	
	bool declareStableParticles(const std::vector<int> &pdgIds);
	bool declareSpecialSettings( const std::vector<std::string> );

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

    string LHEInputFileName;

    /// Switch User Hook flag
    bool            useUserHook;

    std::auto_ptr<LHAupLesHouches>      lhaUP;

	std::auto_ptr<Pythia>	pythia;
	Event			*pythiaEvent;
	HepMC::I_Pythia8	toHepMC;   

};


Pythia8Hadronizer::Pythia8Hadronizer(const edm::ParameterSet &params) :
        BaseHadronizer(params),
	parameters(params.getParameter<edm::ParameterSet>("PythiaParameters")),
	comEnergy(params.getParameter<double>("comEnergy")),
	pythiaPylistVerbosity(params.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
	pythiaHepMCVerbosity(params.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false)),
	maxEventsToPrint(params.getUntrackedParameter<int>("maxEventsToPrint", 0)),
    LHEInputFileName(params.getUntrackedParameter<string>("LHEInputFileName","")),
    useUserHook(false)
{
    if( params.exists( "useUserHook" ) )
      useUserHook = params.getParameter<bool>("useUserHook");
    randomEngine = &getEngineReference();
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
    if(useUserHook) pythia->setUserHooksPtr(new PtHatReweightUserHook());

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

    if(pythiaPylistVerbosity > 10) {
      if(pythiaPylistVerbosity == 11 || pythiaPylistVerbosity == 13)
        pythia->settings.listAll();
      if(pythiaPylistVerbosity == 12 || pythiaPylistVerbosity == 13)
        pythia->particleData.listAll();
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

    if(pythiaPylistVerbosity > 10) {
      if(pythiaPylistVerbosity == 11 || pythiaPylistVerbosity == 13)
        pythia->settings.listAll();
      if(pythiaPylistVerbosity == 12 || pythiaPylistVerbosity == 13)
        pythia->particleData.listAll();
    }

    if(LHEInputFileName != string()) {

      cout << endl;
      cout << "LHE Input File Name = " << LHEInputFileName << endl;
      cout << endl;
      pythia->init(LHEInputFileName);

    } else {

      lhaUP.reset(new LHAupLesHouches());
      lhaUP->loadRunInfo(lheRunInfo());
      pythia->init(lhaUP.get());

    }

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

   for ( size_t i=0; i<pdgIds.size(); i++ )
   {
      // FIXME: need to double check if PID's are the same in Py6 & Py8,
      //        because the HepPDT translation tool is actually for **Py6** 
      // 
      int PyID = HepPID::translatePDTtoPythia( pdgIds[i] ); 
      std::ostringstream pyCard ;
      pyCard << PyID <<":mayDecay=false";
      pythia->readString( pyCard.str() );
      // alternative:
      // set the 2nd input argument warn=false 
      // - this way Py8 will NOT print warnings about unknown particle code(s)
      // pythia->readString( pyCard.str(), false )
   }
   
   return true;

}
bool Pythia8Hadronizer::declareSpecialSettings( const std::vector<std::string> )
{
   return true;
}


void Pythia8Hadronizer::statistics()
{
	pythia->statistics();

	double xsec = pythia->info.sigmaGen(); // cross section in mb
    xsec *= 1.0e9; // translate to pb (CMS/Gen "convention" as of May 2009)
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
    if(LHEInputFileName == string()) {
      //cout << "start loading event" << endl;
      lhaUP->loadEvent(lheEvent());
      //cout << "finish loading event" << endl;
    }

    if (!pythia->next())
        return false;

    // update LHE matching statistics
    //
    lheEvent()->count( lhef::LHERunInfo::kAccepted );

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
  bool lhe = lheEvent() != 0;

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

  // now create the GenEventInfo product from the GenEvent and fill
  // the missing pieces
  eventInfo().reset( new GenEventInfoProduct( event().get() ) );

  // in pythia pthat is used to subdivide samples into different bins
  // in LHE mode the binning is done by the external ME generator
  // which is likely not pthat, so only filling it for Py6 internal mode
  if (!lhe) {
    eventInfo()->setBinningValues(std::vector<double>(1, pythia->info.pTHat()));
  }

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

typedef edm::GeneratorFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8GeneratorFilter;
DEFINE_FWK_MODULE(Pythia8GeneratorFilter);

typedef edm::HadronizerFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8HadronizerFilter;
DEFINE_FWK_MODULE(Pythia8HadronizerFilter);

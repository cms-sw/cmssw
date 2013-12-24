#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <stdint.h>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>

#include <Pythia.h>
#include <HepMCInterface.h>

#include "GeneratorInterface/Pythia8Interface/plugins/RandomP8.h"

#include "GeneratorInterface/Pythia8Interface/plugins/ReweightUserHooks.h"

// PS matchning prototype
//
#include "GeneratorInterface/Pythia8Interface/plugins/JetMatchingHook.h"

// Emission Veto Hook
//
#include "GeneratorInterface/Pythia8Interface/plugins/EmissionVetoHook.h"

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

#include "GeneratorInterface/Pythia8Interface/plugins/LHAupLesHouches.h"

#include "HepPID/ParticleIDTranslations.hh"

#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

using namespace gen;
using namespace Pythia8;

class Pythia8Hadronizer : public BaseHadronizer {
  public:
    Pythia8Hadronizer(const edm::ParameterSet &params);
   ~Pythia8Hadronizer();
 
    bool readSettings( int );
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
    double       comEnergy;
    /// Pythia PYLIST Verbosity flag
    unsigned int pythiaPylistVerbosity;
    /// HepMC verbosity flag
    bool         pythiaHepMCVerbosity;
    /// Events to print if verbosity
    unsigned int maxEventsToPrint;

    string LHEInputFileName;

    std::auto_ptr<LHAupLesHouches>  lhaUP;

    std::auto_ptr<Pythia>   pythia;
    std::auto_ptr<Pythia>   decayer;
    Event*                  pythiaEvent;
    HepMC::I_Pythia8        toHepMC;

    enum { PP, PPbar, ElectronPositron };
    int  fInitialState ; // pp, ppbar, or e-e+

    double fBeam1PZ;
    double fBeam2PZ;

    // Reweight user hook
    //
    UserHooks* fReweightUserHook;
        
    // PS matching protot6ype
    //
    JetMatchingHook* fJetMatchingHook;
	
    // Emission Veto Hook
    //
    EmissionVetoHook* fEmissionVetoHook;

    bool EV_CheckHard;

};


Pythia8Hadronizer::Pythia8Hadronizer(const edm::ParameterSet &params) :
  BaseHadronizer(params),
  parameters(params.getParameter<edm::ParameterSet>("PythiaParameters")),
  comEnergy(params.getParameter<double>("comEnergy")),
  pythiaPylistVerbosity(params.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
  pythiaHepMCVerbosity(params.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false)),
  maxEventsToPrint(params.getUntrackedParameter<int>("maxEventsToPrint", 0)),
  LHEInputFileName(params.getUntrackedParameter<string>("LHEInputFileName","")),
  fInitialState(PP),
  fReweightUserHook(0),
  fJetMatchingHook(0),
  fEmissionVetoHook(0)
{

#ifdef PYTHIA8175
  setenv("PYTHIA8DATA", getenv("PYTHIA8175DATA"), true);
#endif

  randomEngine = &getEngineReference();

  //Old code that used Pythia8 own random engine
  //edm::Service<edm::RandomNumberGenerator> rng;
  //uint32_t seed = rng->mySeed();
  //Pythia8::Rndm::init(seed);

  RandomP8* RP8 = new RandomP8();

  // J.Y.: the following 3 parameters are hacked "for a reason"
  //
  if ( params.exists( "PPbarInitialState" ) )
  {
    if ( fInitialState == PP )
    {
      fInitialState = PPbar;
      edm::LogInfo("GeneratorInterface|Pythia6Interface")
      << "Pythia6 will be initialized for PROTON-ANTIPROTON INITIAL STATE. "
      << "This is a user-request change from the DEFAULT PROTON-PROTON initial state." << std::endl;
      std::cout << "Pythia6 will be initialized for PROTON-ANTIPROTON INITIAL STATE." << std::endl;
      std::cout << "This is a user-request change from the DEFAULT PROTON-PROTON initial state." << std::endl;
    }
    else
    {   
      // probably need to throw on attempt to override ?
    }
  }   
  else if ( params.exists( "ElectronPositronInitialState" ) )
  {
    if ( fInitialState == PP )
    {
      fInitialState = ElectronPositron;
      edm::LogInfo("GeneratorInterface|Pythia6Interface")
      << "Pythia6 will be initialized for ELECTRON-POSITRON INITIAL STATE. "
      << "This is a user-request change from the DEFAULT PROTON-PROTON initial state." << std::endl;
      std::cout << "Pythia6 will be initialized for ELECTRON-POSITRON INITIAL STATE." << std::endl; 
      std::cout << "This is a user-request change from the DEFAULT PROTON-PROTON initial state." << std::endl;
    }
    else
    {   
       // probably need to throw on attempt to override ?
    }
  }
  else if ( params.exists( "ElectronProtonInitialState" ) || params.exists( "PositronProtonInitialState" ) )
  {
    // throw on unknown initial state !
    throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" UNKNOWN INITIAL STATE. \n The allowed initial states are: PP, PPbar, ElectronPositron \n";
  }

  pythia.reset(new Pythia);
  decayer.reset(new Pythia);

  pythia->setRndmEnginePtr(RP8);
  decayer->setRndmEnginePtr(RP8);
    
  if( params.exists( "SLHAFileForPythia8" ) ) {
    std::string slhafilenameshort = params.getParameter<string>("SLHAFileForPythia8");
    edm::FileInPath f1( slhafilenameshort );
    std::string slhafilename = f1.fullPath();
    std::string pythiacommandslha = std::string("SLHA:file = ") + slhafilename;
    pythia->readString(pythiacommandslha);
    for ( ParameterCollector::const_iterator line = parameters.begin();
          line != parameters.end(); ++line ) {
      if (line->find("SLHA:file") != std::string::npos)
        throw cms::Exception("PythiaError") << "Attempted to set SLHA file name twice, "
        << "using Pythia8 card SLHA:file and Pythia8Interface card SLHAFileForPythia8"
        << std::endl;
     }
  } 

  // Reweight user hook
  //
  if( params.exists( "reweightGen" ) )
    fReweightUserHook = new PtHatReweightUserHook();

  // PS matching prototype
  //
  if ( params.exists("jetMatching") )
  {
    edm::ParameterSet jmParams =
      params.getUntrackedParameter<edm::ParameterSet>("jetMatching");
    fJetMatchingHook = new JetMatchingHook( jmParams, &pythia->info );
  }

  // Emission veto
  //
  if ( params.exists("emissionVeto") )
  {   
    EV_CheckHard = false;
    int nversion = (int)(1000.*(pythia->settings.parm("Pythia:versionNumber") - 8.));
    if(nversion > 153) {EV_CheckHard = true;}
    if(params.exists("EV_CheckHard")) EV_CheckHard = params.getParameter<bool>("EV_CheckHard");
    fEmissionVetoHook = new EmissionVetoHook(0, EV_CheckHard);
    pythia->setUserHooksPtr( fEmissionVetoHook );
  }  

  int NHooks=0;
  if(fReweightUserHook) NHooks++;
  if(fJetMatchingHook) NHooks++;
  if(fEmissionVetoHook) NHooks++;
  if(NHooks > 1)
    throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" Too many User Hooks. \n Please choose one from: reweightGen, jetMatching, emissionVeto \n";

  if(fReweightUserHook) pythia->setUserHooksPtr(fReweightUserHook);
  if(fJetMatchingHook) pythia->setUserHooksPtr(fJetMatchingHook);
  if(fEmissionVetoHook) pythia->setUserHooksPtr(fEmissionVetoHook);
}


Pythia8Hadronizer::~Pythia8Hadronizer()
{
// do we need to delete UserHooks/JetMatchingHook here ???

  if(fEmissionVetoHook) {delete fEmissionVetoHook; fEmissionVetoHook=0;}
}


bool Pythia8Hadronizer::readSettings( int )
{
  for ( ParameterCollector::const_iterator line = parameters.begin();
        line != parameters.end(); ++line ) {
    if (line->find("Random:") != std::string::npos)
      throw cms::Exception("PythiaError") << "Attempted to set random number "
        "using Pythia commands. Please use " "the RandomNumberGeneratorService."
        << std::endl;

    if (!pythia->readString(*line)) throw cms::Exception("PythiaError")
			              << "Pythia 8 did not accept \""
				      << *line << "\"." << std::endl;
  }

  if ( pythiaPylistVerbosity > 10 ) {
    if ( pythiaPylistVerbosity == 11 || pythiaPylistVerbosity == 13 )
      pythia->settings.listAll();
    if ( pythiaPylistVerbosity == 12 || pythiaPylistVerbosity == 13 )
      pythia->particleData.listAll();
  }

  return true;
}


bool Pythia8Hadronizer::initializeForInternalPartons()
{

  pythiaEvent = &pythia->event;

  if ( fInitialState == PP ) // default
  {
    pythia->init(2212, 2212, comEnergy);
  }
  else if ( fInitialState == PPbar )
  {
    pythia->init(2212, -2212, comEnergy);
  }
  else if ( fInitialState == ElectronPositron )
  {
    pythia->init(11, -11, comEnergy);
  }    
  else 
  {
    // throw on unknown initial state !
    throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" UNKNOWN INITIAL STATE. \n The allowed initial states are: PP, PPbar, ElectronPositron \n";
  }

  pythia->settings.listChanged();

  return true;
}


bool Pythia8Hadronizer::initializeForExternalPartons()
{

  std::cout << "Initializing for external partons" << std::endl;

  pythiaEvent = &pythia->event;
    
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

  // PS matching prototype
  //
  if ( fJetMatchingHook ) 
  {
    // matcher will be init as well, inside init(...)
    //
    fJetMatchingHook->init ( lheRunInfo() );
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
    // well, actually it looks like Py8 operates in PDT id's rather than Py6's
    //
    // int PyID = HepPID::translatePDTtoPythia( pdgIds[i] ); 
    int PyID = pdgIds[i]; 
    std::ostringstream pyCard ;
    pyCard << PyID <<":mayDecay=false";
    pythia->readString( pyCard.str() );
    // alternative:
    // set the 2nd input argument warn=false 
    // - this way Py8 will NOT print warnings about unknown particle code(s)
    // pythia->readString( pyCard.str(), false )
  }
      
  // init decayer
  decayer->readString("ProcessLevel:all = off"); // The trick!
  decayer->init();
   
  return true;
}


bool Pythia8Hadronizer::declareSpecialSettings( const std::vector<std::string> settings )
{
  for ( unsigned int iss=0; iss<settings.size(); iss++ )
  {
    if ( settings[iss].find("QED-brem-off") == std::string::npos ) continue;
    pythia->readString( "TimeShower:QEDshowerByL=off" );
  }

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
  if (!pythia->next()) return false;

  event().reset(new HepMC::GenEvent);
  toHepMC.fill_next_event(*pythiaEvent, event().get());

  return true;
}


bool Pythia8Hadronizer::hadronize()
{
  if(LHEInputFileName == string()) lhaUP->loadEvent(lheEvent());

  if ( fJetMatchingHook ) 
  {
    fJetMatchingHook->resetMatchingStatus(); 
    fJetMatchingHook->beforeHadronization( lheEvent() );
  }

  bool py8next = pythia->next();
  // if (!pythia->next())
  if (!py8next)
  {
    lheEvent()->count( lhef::LHERunInfo::kSelected );
    event().reset();
    return false;
  }

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

  int NPartsBeforeDecays = pythiaEvent->size();
  int NPartsAfterDecays = event().get()->particles_size();
  int NewBarcode = NPartsAfterDecays;
   
  for ( int ipart=NPartsAfterDecays; ipart>NPartsBeforeDecays; ipart-- )
  {

    HepMC::GenParticle* part = event().get()->barcode_to_particle( ipart );

    if ( part->status() == 1 )
    {
      decayer->event.reset();
      Particle py8part(  part->pdg_id(), 93, 0, 0, 0, 0, 0, 0,
                         part->momentum().x(),
                         part->momentum().y(),
                         part->momentum().z(),
                         part->momentum().t(),
                         part->generated_mass() );
      HepMC::GenVertex* ProdVtx = part->production_vertex();
      py8part.vProd( ProdVtx->position().x(), ProdVtx->position().y(), 
                     ProdVtx->position().z(), ProdVtx->position().t() );
      py8part.tau( (decayer->particleData).tau0( part->pdg_id() ) );
      decayer->event.append( py8part );
      int nentries = decayer->event.size();
      if ( !decayer->event[nentries-1].mayDecay() ) continue;
      decayer->next();
      int nentries1 = decayer->event.size();
      // --> decayer->event.list(std::cout);
      if ( nentries1 <= nentries ) continue; //same number of particles, no decays...
	    
      part->set_status(2);
	    
      Particle& py8daughter = decayer->event[nentries]; // the 1st daughter
      HepMC::GenVertex* DecVtx = new HepMC::GenVertex( HepMC::FourVector(py8daughter.xProd(),
                                                       py8daughter.yProd(),
                                                       py8daughter.zProd(),
                                                       py8daughter.tProd()) );

      DecVtx->add_particle_in( part ); // this will cleanup end_vertex if exists, replace with the new one
                                       // I presume (vtx) barcode will be given automatically
	    
      HepMC::FourVector pmom( py8daughter.px(), py8daughter.py(), py8daughter.pz(), py8daughter.e() );
	    
      HepMC::GenParticle* daughter =
                        new HepMC::GenParticle( pmom, py8daughter.id(), 1 );
	    
      NewBarcode++;
      daughter->suggest_barcode( NewBarcode );
      DecVtx->add_particle_out( daughter );
	    	    
      for ( ipart=nentries+1; ipart<nentries1; ipart++ )
      {
        py8daughter = decayer->event[ipart];
        HepMC::FourVector pmomN( py8daughter.px(), py8daughter.py(), py8daughter.pz(), py8daughter.e() );	    
        HepMC::GenParticle* daughterN =
                        new HepMC::GenParticle( pmomN, py8daughter.id(), 1 );
        NewBarcode++;
        daughterN->suggest_barcode( NewBarcode );
        DecVtx->add_particle_out( daughterN );
      }
	    
      event().get()->add_vertex( DecVtx );

    }
 } 
   
 return true;
}


void Pythia8Hadronizer::finalizeEvent()
{
  bool lhe = lheEvent() != 0;

  event()->set_signal_process_id(pythia->info.code());
  event()->set_event_scale(pythia->info.pTHat());	//FIXME

  //cout.precision(10);
  //cout << " pt = " << pythia->info.pTHat() << " weights = "
  //     << pythia->info.weight() << " "
  //     << fReweightUserHook->biasedSelectionWeight() << endl;

  if (event()->alphaQED() <= 0)
    event()->set_alphaQED( pythia->info.alphaEM() );
  if (event()->alphaQCD() <= 0)
    event()->set_alphaQCD( pythia->info.alphaS() );

  HepMC::GenCrossSection xsec;
  xsec.set_cross_section( pythia->info.sigmaGen() * 1e9,
                          pythia->info.sigmaErr() * 1e9);
  event()->set_cross_section(xsec);

  // Putting pdf info into the HepMC record
  // There is the overloaded pythia8 HepMCInterface method fill_next_event
  // that does this, but CMSSW GeneratorInterface does not fill HepMC
  // record according to the HepMC convention (stores f(x) instead of x*f(x)
  // and converts gluon PDG ID to zero). For this reason we use the
  // method fill_next_event (above) that does NOT this and fill pdf info here
  //
  int id1 = pythia->info.id1();
  int id2 = pythia->info.id2();
  if (id1 == 21) id1 = 0;
  if (id2 == 21) id2 = 0;
  double x1 = pythia->info.x1();
  double x2 = pythia->info.x2();
  //double Q = pythia->info.QRen();
  double Q = pythia->info.QFac();
  double pdf1 = pythia->info.pdf1() / pythia->info.x1();
  double pdf2 = pythia->info.pdf2() / pythia->info.x2();
  event()->set_pdf_info(HepMC::PdfInfo(id1,id2,x1,x2,Q,pdf1,pdf2));

  // Storing weights. Will be moved to pythia8 HepMCInterface
  //
  if (lhe && std::abs(lheRunInfo()->getHEPRUP()->IDWTUP) == 4)
    // translate mb to pb (CMS/Gen "convention" as of May 2009)
    event()->weights().push_back( pythia->info.weight() * 1.0e9 );
  else
    event()->weights().push_back( pythia->info.weight() );

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

#ifdef PYTHIA8175

typedef edm::GeneratorFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8175GeneratorFilter;
DEFINE_FWK_MODULE(Pythia8175GeneratorFilter);

typedef edm::HadronizerFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8175HadronizerFilter;
DEFINE_FWK_MODULE(Pythia8175HadronizerFilter);

#else

typedef edm::GeneratorFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8GeneratorFilter;
DEFINE_FWK_MODULE(Pythia8GeneratorFilter);

typedef edm::HadronizerFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8HadronizerFilter;
DEFINE_FWK_MODULE(Pythia8HadronizerFilter);

#endif

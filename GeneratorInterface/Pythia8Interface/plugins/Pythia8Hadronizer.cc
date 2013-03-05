#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <stdint.h>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>

#include <Pythia.h>
#include <HepMCInterface.h>

#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"

#include "GeneratorInterface/Pythia8Interface/plugins/ReweightUserHooks.h"

// PS matchning prototype
//
#include "GeneratorInterface/Pythia8Interface/plugins/JetMatchingHook.h"


// Emission Veto Hooks
//
#include "GeneratorInterface/Pythia8Interface/plugins/EmissionVetoHook.h"
#include "GeneratorInterface/Pythia8Interface/plugins/EmissionVetoHook1.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"

#include "GeneratorInterface/Pythia8Interface/plugins/LHAupLesHouches.h"

#include "HepPID/ParticleIDTranslations.hh"

#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

using namespace gen;
using namespace Pythia8;

class Pythia8Hadronizer : public BaseHadronizer, public Py8InterfaceBase {

  public:

    Pythia8Hadronizer(const edm::ParameterSet &params);
   ~Pythia8Hadronizer();
 
    bool initializeForInternalPartons();
    bool initializeForExternalPartons();
	
    bool generatePartonsAndHadronize();
    bool hadronize();
    void finalizeEvent();

    void statistics();

    const char *classname() const { return "Pythia8Hadronizer"; }

  private:

    /// Center-of-Mass energy
    double       comEnergy;

    string LHEInputFileName;
    std::auto_ptr<LHAupLesHouches>  lhaUP;

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
	
    // Emission Veto Hooks
    //
    EmissionVetoHook* fEmissionVetoHook;
    EmissionVetoHook1* fEmissionVetoHook1;

    int  EV1_nFinal;
    bool EV1_vetoOn;
    int  EV1_maxVetoCount;
    int  EV1_pThardMode;
    int  EV1_pTempMode;
    int  EV1_emittedMode;
    int  EV1_pTdefMode;
    bool EV1_MPIvetoOn;

};


Pythia8Hadronizer::Pythia8Hadronizer(const edm::ParameterSet &params) :
  BaseHadronizer(params), Py8InterfaceBase(params),
  comEnergy(params.getParameter<double>("comEnergy")),
  LHEInputFileName(params.getUntrackedParameter<string>("LHEInputFileName","")),
  fInitialState(PP),
  fReweightUserHook(0),
  fJetMatchingHook(0),
  fEmissionVetoHook(0),fEmissionVetoHook1(0)
{

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
    
  if( params.exists( "SLHAFileForPythia8" ) ) {
    std::string slhafilenameshort = params.getParameter<string>("SLHAFileForPythia8");
    edm::FileInPath f1( slhafilenameshort );
    std::string slhafilename = f1.fullPath();
    std::string pythiacommandslha = std::string("SLHA:file = ") + slhafilename;
    fMasterGen->readString(pythiacommandslha);
    for ( ParameterCollector::const_iterator line = fParameters.begin();
          line != fParameters.end(); ++line ) {
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
      std::string scheme = jmParams.getParameter<std::string>("scheme");
      if ( scheme == "Madgraph" || scheme == "MadgraphFastJet" )
      {
         fJetMatchingHook = new JetMatchingHook( jmParams, &fMasterGen->info );
      }
  }

  // Emission vetos
  //
  if ( params.exists("emissionVeto") )
  {   
    fEmissionVetoHook = new EmissionVetoHook(0);
  }

  if ( params.exists("emissionVeto1") )
  {
    EV1_nFinal = -1;
    if(params.exists("EV1_nFinal")) EV1_nFinal = params.getParameter<int>("EV1_nFinal");
    EV1_vetoOn = true;
    if(params.exists("EV1_vetoOn")) EV1_vetoOn = params.getParameter<bool>("EV1_vetoOn");
    EV1_maxVetoCount = 10;
    if(params.exists("EV1_maxVetoCount")) EV1_maxVetoCount = params.getParameter<int>("EV1_maxVetoCount");
    EV1_pThardMode = 1;
    if(params.exists("EV1_pThardMode")) EV1_pThardMode = params.getParameter<int>("EV1_pThardMode");
    EV1_pTempMode = 0;
    if(params.exists("EV1_pTempMode")) EV1_pTempMode = params.getParameter<int>("EV1_pTempMode");
    if(EV1_pTempMode > 2 || EV1_pTempMode < 0)
      throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
        <<" Wrong value for EV1_pTempMode code\n";
    EV1_emittedMode = 0;
    if(params.exists("EV1_emittedMode")) EV1_emittedMode = params.getParameter<int>("EV1_emittedMode");
    EV1_pTdefMode = 1;
    if(params.exists("EV1_pTdefMode")) EV1_pTdefMode = params.getParameter<int>("EV1_pTdefMode");
    EV1_MPIvetoOn = false;
    if(params.exists("EV1_MPIvetoOn")) EV1_MPIvetoOn = params.getParameter<bool>("EV1_MPIvetoOn");
    fEmissionVetoHook1 = new EmissionVetoHook1(EV1_nFinal, EV1_vetoOn, 
                               EV1_maxVetoCount, EV1_pThardMode, EV1_pTempMode,
                               EV1_emittedMode, EV1_pTdefMode, EV1_MPIvetoOn, 0);
  }

  int NHooks=0;
  if(fReweightUserHook) NHooks++;
  if(fJetMatchingHook) NHooks++;
  if(fEmissionVetoHook) NHooks++;
  if(fEmissionVetoHook1) NHooks++;
  if(NHooks > 1)
    throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" Too many User Hooks. \n Please choose one from: reweightGen, jetMatching, emissionVeto \n";

  if(fReweightUserHook) fMasterGen->setUserHooksPtr(fReweightUserHook);
  if(fJetMatchingHook) fMasterGen->setUserHooksPtr(fJetMatchingHook);
  if(fEmissionVetoHook || fEmissionVetoHook1) {
    cout << "Turning on Emission Veto Hook";
    if(fEmissionVetoHook1) cout << " 1";
    cout << endl;
    int nversion = (int)(1000.*(fMasterGen->settings.parm("Pythia:versionNumber") - 8.));
    if(nversion < 157) {
      cout << "obsolete pythia8 version for this Emission Veto code" << endl;
      cout << "Please update pythia8 version using the instructions here:" << endl;
      cout << "https://twiki.cern.ch/twiki/bin/view/CMS/Pythia8Interface" << endl;
      cout << "or try to use tag V00-01-28 of this interface" << endl;
      throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
        <<" Obsolete pythia8 version for this Emission Veto code\n";
    }
    if(fEmissionVetoHook) fMasterGen->setUserHooksPtr(fEmissionVetoHook);
    if(fEmissionVetoHook1) fMasterGen->setUserHooksPtr(fEmissionVetoHook1);
  }
}


Pythia8Hadronizer::~Pythia8Hadronizer()
{
// do we need to delete UserHooks/JetMatchingHook here ???

  if(fEmissionVetoHook) {delete fEmissionVetoHook; fEmissionVetoHook=0;}
  if(fEmissionVetoHook1) {delete fEmissionVetoHook1; fEmissionVetoHook1=0;}
}

bool Pythia8Hadronizer::initializeForInternalPartons()
{

  // pythiaEvent = &pythia->event;

  if ( fInitialState == PP ) // default
  {
    fMasterGen->init(2212, 2212, comEnergy);
  }
  else if ( fInitialState == PPbar )
  {
    fMasterGen->init(2212, -2212, comEnergy);
  }
  else if ( fInitialState == ElectronPositron )
  {
    fMasterGen->init(11, -11, comEnergy);
  }    
  else 
  {
    // throw on unknown initial state !
    throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" UNKNOWN INITIAL STATE. \n The allowed initial states are: PP, PPbar, ElectronPositron \n";
  }

  fMasterGen->settings.listChanged();

  return true;
}


bool Pythia8Hadronizer::initializeForExternalPartons()
{

  std::cout << "Initializing for external partons" << std::endl;

  // pythiaEvent = &pythia->event;
    
  if(LHEInputFileName != string()) {

    cout << endl;
    cout << "LHE Input File Name = " << LHEInputFileName << endl;
    cout << endl;
    fMasterGen->init(LHEInputFileName);

  } else {

    lhaUP.reset(new LHAupLesHouches());
    lhaUP->loadRunInfo(lheRunInfo());
    
    if ( fJetMatchingHook )
    {
       fJetMatchingHook->init ( lheRunInfo() );
    }
    
    fMasterGen->init(lhaUP.get());

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


void Pythia8Hadronizer::statistics()
{
  fMasterGen->statistics();

  double xsec = fMasterGen->info.sigmaGen(); // cross section in mb
  xsec *= 1.0e9; // translate to pb (CMS/Gen "convention" as of May 2009)
  runInfo().setInternalXSec(xsec);
}


bool Pythia8Hadronizer::generatePartonsAndHadronize()
{

  if (!fMasterGen->next()) return false;

  event().reset(new HepMC::GenEvent);
  toHepMC.fill_next_event( fMasterGen->event, event().get());

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

  bool py8next = fMasterGen->next();

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
  toHepMC.fill_next_event( fMasterGen->event, event().get());

  return true;
}


void Pythia8Hadronizer::finalizeEvent()
{
  bool lhe = lheEvent() != 0;

  event()->set_signal_process_id(fMasterGen->info.code());
  event()->set_event_scale(fMasterGen->info.pTHat());	//FIXME

  //cout.precision(10);
  //cout << " pt = " << fMasterGen->info.pTHat() << " weights = "
  //     << fMasterGen->info.weight() << " "
  //     << fReweightUserHook->biasedSelectionWeight() << endl;

  if (event()->alphaQED() <= 0)
    event()->set_alphaQED( fMasterGen->info.alphaEM() );
  if (event()->alphaQCD() <= 0)
    event()->set_alphaQCD( fMasterGen->info.alphaS() );

  HepMC::GenCrossSection xsec;
  xsec.set_cross_section( fMasterGen->info.sigmaGen() * 1e9,
                          fMasterGen->info.sigmaErr() * 1e9);
  event()->set_cross_section(xsec);

  // Putting pdf info into the HepMC record
  // There is the overloaded pythia8 HepMCInterface method fill_next_event
  // that does this, but CMSSW GeneratorInterface does not fill HepMC
  // record according to the HepMC convention (stores f(x) instead of x*f(x)
  // and converts gluon PDG ID to zero). For this reason we use the
  // method fill_next_event (above) that does NOT this and fill pdf info here
  //
  int id1 = fMasterGen->info.id1();
  int id2 = fMasterGen->info.id2();
  if (id1 == 21) id1 = 0;
  if (id2 == 21) id2 = 0;
  double x1 = fMasterGen->info.x1();
  double x2 = fMasterGen->info.x2();
  //double Q = fMasterGen->info.QRen();
  double Q = fMasterGen->info.QFac();
  double pdf1 = fMasterGen->info.pdf1() / fMasterGen->info.x1();
  double pdf2 = fMasterGen->info.pdf2() / fMasterGen->info.x2();
  event()->set_pdf_info(HepMC::PdfInfo(id1,id2,x1,x2,Q,pdf1,pdf2));

  // Storing weights. Will be moved to pythia8 HepMCInterface
  //
  if (lhe && std::abs(lheRunInfo()->getHEPRUP()->IDWTUP) == 4)
    // translate mb to pb (CMS/Gen "convention" as of May 2009)
    event()->weights().push_back( fMasterGen->info.weight() * 1.0e9 );
  else
    event()->weights().push_back( fMasterGen->info.weight() );

  // now create the GenEventInfo product from the GenEvent and fill
  // the missing pieces
  eventInfo().reset( new GenEventInfoProduct( event().get() ) );

  // in pythia pthat is used to subdivide samples into different bins
  // in LHE mode the binning is done by the external ME generator
  // which is likely not pthat, so only filling it for Py6 internal mode
  if (!lhe) {
    eventInfo()->setBinningValues(std::vector<double>(1, fMasterGen->info.pTHat()));
  }

  //******** Verbosity ********

  if (maxEventsToPrint > 0 &&
      (pythiaPylistVerbosity || pythiaHepMCVerbosity)) {
    maxEventsToPrint--;
    if (pythiaPylistVerbosity) {
      fMasterGen->info.list(std::cout); 
      fMasterGen->event.list(std::cout);
    } 

    if (pythiaHepMCVerbosity) {
      std::cout << "Event process = "
                << fMasterGen->info.code() << "\n"
                << "----------------------" << std::endl;
      event()->print();
    }
  }
}


typedef edm::GeneratorFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8GeneratorFilter;
DEFINE_FWK_MODULE(Pythia8GeneratorFilter);


typedef edm::HadronizerFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8HadronizerFilter;
DEFINE_FWK_MODULE(Pythia8HadronizerFilter);

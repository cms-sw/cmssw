#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <stdint.h>
#include <vector>

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"
#include "GeneratorInterface/Pythia8Interface/plugins/HepMCA2.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"

#include "GeneratorInterface/Pythia8Interface/plugins/ReweightUserHooks.h"

// PS matchning prototype
//
#include "GeneratorInterface/Pythia8Interface/plugins/JetMatchingHook.h"
#include "Pythia8Plugins/JetMatching.h"
#include "Pythia8Plugins/aMCatNLOHooks.h"

// Emission Veto Hooks
//
#include "Pythia8Plugins/PowhegHooks.h"
#include "GeneratorInterface/Pythia8Interface/plugins/EmissionVetoHook1.h"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
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

namespace CLHEP {
  class HepRandomEngine;
}

using namespace gen;
using namespace Pythia8;

class Pythia8Hadronizer : public BaseHadronizer, public Py8InterfaceBase {

  public:

    Pythia8Hadronizer(const edm::ParameterSet &params);
   ~Pythia8Hadronizer();
 
    bool initializeForInternalPartons() override;
    bool initializeForExternalPartons();
	
    bool generatePartonsAndHadronize() override;
    bool hadronize();

    virtual bool residualDecay();

    void finalizeEvent() override;

    void statistics() override;

    const char *classname() const override { return "Pythia8Hadronizer"; }

  private:

    virtual void doSetRandomEngine(CLHEP::HepRandomEngine* v) override { p8SetRandomEngine(v); }
    virtual std::vector<std::string> const& doSharedResources() const override { return p8SharedResources; }

    /// Center-of-Mass energy
    double       comEnergy;

    string LHEInputFileName;
    std::auto_ptr<LHAupLesHouches>  lhaUP;

    enum { PP, PPbar, ElectronPositron };
    int  fInitialState ; // pp, ppbar, or e-e+

    double fBeam1PZ;
    double fBeam2PZ;

    // Reweight user hooks
    //
    UserHooks* fReweightUserHook;
    UserHooks* fReweightRapUserHook;  
    UserHooks* fReweightPtHatRapUserHook;
        
    // PS matching prototype
    //
    JetMatchingHook* fJetMatchingHook;
    Pythia8::JetMatchingMadgraph *fJetMatchingPy8InternalHook;
    Pythia8::amcnlo_unitarised_interface *fMergingHook;
    
    // Emission Veto Hooks
    //
    PowhegHooks* fEmissionVetoHook;
    EmissionVetoHook1* fEmissionVetoHook1;
    
    int  EV1_nFinal;
    bool EV1_vetoOn;
    int  EV1_maxVetoCount;
    int  EV1_pThardMode;
    int  EV1_pTempMode;
    int  EV1_emittedMode;
    int  EV1_pTdefMode;
    bool EV1_MPIvetoOn;   

    static const std::vector<std::string> p8SharedResources;
    
    std::string slhafile_;

    vector<float> DJR;
    int nME;
    int nMEFiltered;

    int nISRveto;
    int nFSRveto;

    int NHooks;

};

const std::vector<std::string> Pythia8Hadronizer::p8SharedResources = { edm::SharedResourceNames::kPythia8 };

Pythia8Hadronizer::Pythia8Hadronizer(const edm::ParameterSet &params) :
  BaseHadronizer(params), Py8InterfaceBase(params),
  comEnergy(params.getParameter<double>("comEnergy")),
  LHEInputFileName(params.getUntrackedParameter<string>("LHEInputFileName","")),
  fInitialState(PP),
  fReweightUserHook(0),fReweightRapUserHook(0),fReweightPtHatRapUserHook(0),
  fJetMatchingHook(0),fJetMatchingPy8InternalHook(0), fMergingHook(0),
  fEmissionVetoHook(0), fEmissionVetoHook1(0), nME(-1), nMEFiltered(-1), nISRveto(0), nFSRveto(0),
  NHooks(0)
{

  // J.Y.: the following 3 parameters are hacked "for a reason"
  //
  if ( params.exists( "PPbarInitialState" ) )
  {
    if ( fInitialState == PP )
    {
      fInitialState = PPbar;
      edm::LogImportant("GeneratorInterface|Pythia8Interface")
      << "Pythia8 will be initialized for PROTON-ANTIPROTON INITIAL STATE. "
      << "This is a user-request change from the DEFAULT PROTON-PROTON initial state.";
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
      edm::LogInfo("GeneratorInterface|Pythia8Interface")
      << "Pythia8 will be initialized for ELECTRON-POSITRON INITIAL STATE. "
      << "This is a user-request change from the DEFAULT PROTON-PROTON initial state.";
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
    
    fMasterGen->settings.mode("SLHA:readFrom", 2);
    fMasterGen->settings.word("SLHA:file", f1.fullPath());    
    
    for ( ParameterCollector::const_iterator line = fParameters.begin();
          line != fParameters.end(); ++line ) {
      if (line->find("SLHA:file") != std::string::npos)
        throw cms::Exception("PythiaError") << "Attempted to set SLHA file name twice, "
        << "using Pythia8 card SLHA:file and Pythia8Interface card SLHAFileForPythia8"
        << std::endl;
     }  
  }
  else if( params.exists( "SLHATableForPythia8" ) ) {
    std::string slhatable = params.getParameter<string>("SLHATableForPythia8");
        
    char tempslhaname[] = "pythia8SLHAtableXXXXXX";
    int fd = mkstemp(tempslhaname);
    write(fd,slhatable.c_str(),slhatable.size());
    close(fd);
    
    slhafile_ = tempslhaname;
    
    fMasterGen->settings.mode("SLHA:readFrom", 2);
    fMasterGen->settings.word("SLHA:file", slhafile_);    
    
    for ( ParameterCollector::const_iterator line = fParameters.begin();
          line != fParameters.end(); ++line ) {
      if (line->find("SLHA:file") != std::string::npos)
        throw cms::Exception("PythiaError") << "Attempted to set SLHA file name twice, "
        << "using Pythia8 card SLHA:file and Pythia8Interface card SLHATableForPythia8"
        << std::endl;
     }  
  }

  // Reweight user hook
  //
  if( params.exists( "reweightGen" ) )
    fReweightUserHook = new PtHatReweightUserHook();
  if( params.exists( "reweightGenRap" ) )
  {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGenRap";
    edm::ParameterSet rgrParams =
      params.getParameter<edm::ParameterSet>("reweightGenRap");
    fReweightRapUserHook =
      new RapReweightUserHook(rgrParams.getParameter<std::string>("yLabSigmaFunc"),
                              rgrParams.getParameter<double>("yLabPower"),
                              rgrParams.getParameter<std::string>("yCMSigmaFunc"),
                              rgrParams.getParameter<double>("yCMPower"),
                              rgrParams.getParameter<double>("pTHatMin"),
                              rgrParams.getParameter<double>("pTHatMax"));
    edm::LogInfo("Pythia8Interface") << "End setup for reweightGenRap";
  }
  if( params.exists( "reweightGenPtHatRap" ) )
  {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGenPtHatRap";
    edm::ParameterSet rgrParams =
      params.getParameter<edm::ParameterSet>("reweightGenPtHatRap");
    fReweightPtHatRapUserHook =
      new PtHatRapReweightUserHook(rgrParams.getParameter<std::string>("yLabSigmaFunc"),
                                   rgrParams.getParameter<double>("yLabPower"),
                                   rgrParams.getParameter<std::string>("yCMSigmaFunc"),
                                   rgrParams.getParameter<double>("yCMPower"),
                                   rgrParams.getParameter<double>("pTHatMin"),
                                   rgrParams.getParameter<double>("pTHatMax"));
    edm::LogInfo("Pythia8Interface") << "End setup for reweightGenPtHatRap";
  }

  if( params.exists( "useUserHook" ) )
    throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" Obsolete parameter: useUserHook \n Please use the actual one instead \n";

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

  // Pythia8Interface emission veto
  //
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

  if(fReweightUserHook) NHooks++;
  if(fReweightRapUserHook) NHooks++;
  if(fReweightPtHatRapUserHook) NHooks++;
  if(fJetMatchingHook) NHooks++;
  if(fEmissionVetoHook1) NHooks++;
  if(NHooks > 1)
    throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" Too many User Hooks. \n Please choose one from: reweightGen, reweightGenRap, reweightGenPtHatRap, jetMatching, emissionVeto1 \n";
  if(fReweightUserHook) fMasterGen->setUserHooksPtr(fReweightUserHook);
  if(fReweightRapUserHook) fMasterGen->setUserHooksPtr(fReweightRapUserHook);
  if(fReweightPtHatRapUserHook) fMasterGen->setUserHooksPtr(fReweightPtHatRapUserHook);
  if(fJetMatchingHook) fMasterGen->setUserHooksPtr(fJetMatchingHook);
  if(fEmissionVetoHook1) { 
    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook 1 from CMSSW Pythia8Interface";
    fMasterGen->setUserHooksPtr(fEmissionVetoHook1);
  }

}


Pythia8Hadronizer::~Pythia8Hadronizer()
{
// do we need to delete UserHooks/JetMatchingHook here ???
  if(fEmissionVetoHook) {delete fEmissionVetoHook; fEmissionVetoHook=0;}
  if(fEmissionVetoHook1) {delete fEmissionVetoHook1; fEmissionVetoHook1=0;}
  
  //clean up temp file
  if (!slhafile_.empty()) {
    std::remove(slhafile_.c_str());
  }
  
}

bool Pythia8Hadronizer::initializeForInternalPartons()
{
  
  bool status = false, status1 = false;
  
  if ( fInitialState == PP ) // default
  {
    fMasterGen->settings.mode("Beams:idA", 2212);
    fMasterGen->settings.mode("Beams:idB", 2212);
  }
  else if ( fInitialState == PPbar )
  {
    fMasterGen->settings.mode("Beams:idA", 2212);
    fMasterGen->settings.mode("Beams:idB", -2212);
  }
  else if ( fInitialState == ElectronPositron )
  {
    fMasterGen->settings.mode("Beams:idA", 11);
    fMasterGen->settings.mode("Beams:idB", -11);
  }    
  else 
  {
    // throw on unknown initial state !
    throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" UNKNOWN INITIAL STATE. \n The allowed initial states are: PP, PPbar, ElectronPositron \n";
  }

  fMasterGen->settings.parm("Beams:eCM", comEnergy);
  edm::LogInfo("Pythia8Interface") << "Initializing MasterGen";
  status = fMasterGen->init();

  if ( pythiaPylistVerbosity > 10 )
  {
    if ( pythiaPylistVerbosity == 11 || pythiaPylistVerbosity == 13 )
           fMasterGen->settings.listAll();
    if ( pythiaPylistVerbosity == 12 || pythiaPylistVerbosity == 13 )
           fMasterGen->particleData.listAll();
  }

  // init decayer
  fDecayer->settings.flag("ProcessLevel:all", false ); // trick
  fDecayer->settings.flag("ProcessLevel:resonanceDecays", true );
  edm::LogInfo("Pythia8Interface") << "Initializing Decayer";
  status1 = fDecayer->init();

  return (status&&status1);
}


bool Pythia8Hadronizer::initializeForExternalPartons()
{

  edm::LogInfo("Pythia8Interface") << "Initializing for external partons";

  bool status = false, status1 = false;
  
  if((fMasterGen->settings.mode("POWHEG:veto") > 0 || fMasterGen->settings.mode("POWHEG:MPIveto") > 0) && !fEmissionVetoHook) {

    if(fJetMatchingHook || fEmissionVetoHook1)
      throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" Attempt to turn on PowhegHooks by pythia8 settings but there are incompatible hooks on \n Incompatible are : jetMatching, emissionVeto1 \n";

    fEmissionVetoHook = new PowhegHooks();

    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook from pythia8 code";
    fMasterGen->setUserHooksPtr(fEmissionVetoHook);

  }

  //adapted from main89.cc in pythia8 examples
  bool internalMatching = fMasterGen->settings.flag("JetMatching:merge");
  bool internalMerging = !(fMasterGen->settings.word("Merging:Process").compare("void")==0);
  
  if (internalMatching && internalMerging) {
    throw edm::Exception(edm::errors::Configuration,"Pythia8Interface")
      <<" Only one jet matching/merging scheme can be used at a time. \n";
  }
  
  if (internalMatching && !fJetMatchingPy8InternalHook) {
    fJetMatchingPy8InternalHook = new Pythia8::JetMatchingMadgraph;
    fMasterGen->setUserHooksPtr(fJetMatchingPy8InternalHook);
  }
  
  if (internalMerging && !fMergingHook) {
    int scheme = ( fMasterGen->settings.flag("Merging:doUMEPSTree")
                || fMasterGen->settings.flag("Merging:doUMEPSSubt")) ?
                1 :
                 ( ( fMasterGen->settings.flag("Merging:doUNLOPSTree")
                || fMasterGen->settings.flag("Merging:doUNLOPSSubt")
                || fMasterGen->settings.flag("Merging:doUNLOPSLoop")
                || fMasterGen->settings.flag("Merging:doUNLOPSSubtNLO")) ?
                2 :
                0 );
    fMergingHook = new Pythia8::amcnlo_unitarised_interface(scheme);
    fMasterGen->setUserHooksPtr(fMergingHook);
  }
  
  
  if(LHEInputFileName != string()) {

    edm::LogInfo("Pythia8Interface") << "Initialize direct pythia8 reading from LHE file "
                                     << LHEInputFileName;
    edm::LogInfo("Pythia8Interface") << "Some LHE information can be not stored";
    fMasterGen->settings.mode("Beams:frameType", 4);
    fMasterGen->settings.word("Beams:LHEF", LHEInputFileName);
    status = fMasterGen->init();

  } else {

    lhaUP.reset(new LHAupLesHouches());
    lhaUP->setScalesFromLHEF(fMasterGen->settings.flag("Beams:setProductionScalesFromLHEF"));
    lhaUP->loadRunInfo(lheRunInfo());
    
    if ( fJetMatchingHook )
    {
       fJetMatchingHook->init ( lheRunInfo() );
    }
    
    fMasterGen->settings.mode("Beams:frameType", 5);
    fMasterGen->setLHAupPtr(lhaUP.get());
    edm::LogInfo("Pythia8Interface") << "Initializing MasterGen";
    status = fMasterGen->init();
  }
  
  if ( pythiaPylistVerbosity > 10 )
  {
    if ( pythiaPylistVerbosity == 11 || pythiaPylistVerbosity == 13 )
           fMasterGen->settings.listAll();
    if ( pythiaPylistVerbosity == 12 || pythiaPylistVerbosity == 13 )
           fMasterGen->particleData.listAll();
  }

  // init decayer
  fDecayer->settings.flag("ProcessLevel:all", false ); // trick
  fDecayer->settings.flag("ProcessLevel:resonanceDecays", true );
  edm::LogInfo("Pythia8Interface") << "Initializing Decayer";
  status1 = fDecayer->init();

  return (status&&status1);
}


void Pythia8Hadronizer::statistics()
{
  fMasterGen->stat();

  if(fEmissionVetoHook) {
    edm::LogPrint("Pythia8Interface") << "\n"
      << "Number of ISR vetoed = " << nISRveto;
    edm::LogPrint("Pythia8Interface")
      << "Number of FSR vetoed = " << nFSRveto;
  }

  double xsec = fMasterGen->info.sigmaGen(); // cross section in mb
  xsec *= 1.0e9; // translate to pb (CMS/Gen "convention" as of May 2009)
  double err  = fMasterGen->info.sigmaErr(); // cross section err in mb
  err  *= 1.0e9; // translate to pb (CMS/Gen "convention" as of May 2009)
  runInfo().setInternalXSec(GenRunInfoProduct::XSec(xsec,err));
}


bool Pythia8Hadronizer::generatePartonsAndHadronize()
{

  if (!fMasterGen->next()) return false;

  event().reset(new HepMC::GenEvent);
  return toHepMC.fill_next_event( *(fMasterGen.get()), event().get());

}


bool Pythia8Hadronizer::hadronize()
{
  DJR.resize(0);
  nME = -1;
  nMEFiltered = -1;
  if(LHEInputFileName == string()) lhaUP->loadEvent(lheEvent());

  if ( fJetMatchingHook ) 
  {
    fJetMatchingHook->resetMatchingStatus(); 
    fJetMatchingHook->beforeHadronization( lheEvent() );
  }

  bool py8next = fMasterGen->next();

  double mergeweight = fMasterGen.get()->info.mergingWeightNLO();
  if (fMergingHook) {
    mergeweight *= fMergingHook->getNormFactor();
  }
  
  
  //protect against 0-weight from ckkw or similar
  if (!py8next || std::abs(mergeweight)==0.)
  {
    lheEvent()->count( lhef::LHERunInfo::kSelected, 1.0, mergeweight );
    event().reset();
    return false;
  }
  
  if (fJetMatchingPy8InternalHook) {
    const std::vector<double> djrmatch = fJetMatchingPy8InternalHook->GetDJR();
    //cap size of djr vector to save storage space (keep only up to first 6 elements)
    unsigned int ndjr = std::min(djrmatch.size(), std::vector<double>::size_type(6));
    for (unsigned int idjr=0; idjr<ndjr; ++idjr) {
      DJR.push_back(djrmatch[idjr]);
    }
    
    nME=fJetMatchingPy8InternalHook->nMEpartons().first;
    nMEFiltered=fJetMatchingPy8InternalHook->nMEpartons().second;
  }
  
  // update LHE matching statistics
  //
  lheEvent()->count( lhef::LHERunInfo::kAccepted, 1.0, mergeweight );

  event().reset(new HepMC::GenEvent);
  bool py8hepmc =  toHepMC.fill_next_event( *(fMasterGen.get()), event().get());
  if (!py8hepmc) {
    return false;
  }
  
  //add ckkw/umeps/unlops merging weight
  if (mergeweight!=1.) {
    event()->weights()[0] *= mergeweight;
  }
  
  if (fEmissionVetoHook) {
    nISRveto += fEmissionVetoHook->getNISRveto();
    nFSRveto += fEmissionVetoHook->getNFSRveto();  
  }

  return true;
  
  
}


bool Pythia8Hadronizer::residualDecay()
{

  Event* pythiaEvent = &(fMasterGen->event);

  int NPartsBeforeDecays = pythiaEvent->size();
  int NPartsAfterDecays = event().get()->particles_size();

  if(NPartsAfterDecays == NPartsBeforeDecays) return true;

  HepMC::Pythia8ToHepMCA toHepMCA;
  bool result = true;
 
  for ( int ipart=NPartsAfterDecays; ipart>NPartsBeforeDecays; ipart-- )
  {

    HepMC::GenParticle* part = event().get()->barcode_to_particle( ipart );

    if ( part->status() == 1 && (fDecayer->particleData).canDecay(part->pdg_id()) )
    {
      fDecayer->event.reset();
      Particle py8part(  part->pdg_id(), 93, 0, 0, 0, 0, 0, 0,
                         part->momentum().x(),
                         part->momentum().y(),
                         part->momentum().z(),
                         part->momentum().t(),
                         part->generated_mass() );
      HepMC::GenVertex* ProdVtx = part->production_vertex();
      py8part.vProd( ProdVtx->position().x(), ProdVtx->position().y(),
                     ProdVtx->position().z(), ProdVtx->position().t() );
      py8part.tau( (fDecayer->particleData).tau0( part->pdg_id() ) );
      fDecayer->event.append( py8part );
      int nentries = fDecayer->event.size();
      if ( !fDecayer->event[nentries-1].mayDecay() ) continue;
      fDecayer->next();
      int nentries1 = fDecayer->event.size();
      if ( nentries1 <= nentries ) continue; //same number of particles, no decays...

      part->set_status(2);

      result = toHepMCA.append_event( fDecayer->event, event().get(), part);

    }
  }

  return result;

}


void Pythia8Hadronizer::finalizeEvent()
{
  bool lhe = lheEvent() != 0;

  // now create the GenEventInfo product from the GenEvent and fill
  // the missing pieces
  eventInfo().reset( new GenEventInfoProduct( event().get() ) );

  // in pythia pthat is used to subdivide samples into different bins
  // in LHE mode the binning is done by the external ME generator
  // which is likely not pthat, so only filling it for Py6 internal mode
  if (!lhe) {
    eventInfo()->setBinningValues(std::vector<double>(1, fMasterGen->info.pTHat()));
  }
  
  eventInfo()->setDJR(DJR);
  eventInfo()->setNMEPartons(nME);
  eventInfo()->setNMEPartonsFiltered(nMEFiltered);

  //******** Verbosity ********

  if (maxEventsToPrint > 0 &&
      (pythiaPylistVerbosity || pythiaHepMCVerbosity ||
                                pythiaHepMCVerbosityParticles) ) {
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
    if (pythiaHepMCVerbosityParticles) {
      std::cout << "Event process = "
                << fMasterGen->info.code() << "\n"
                << "----------------------" << std::endl;
      ascii_io->write_event(event().get());
    }
  }
}

typedef edm::GeneratorFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8GeneratorFilter;
DEFINE_FWK_MODULE(Pythia8GeneratorFilter);


typedef edm::HadronizerFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8HadronizerFilter;
DEFINE_FWK_MODULE(Pythia8HadronizerFilter);

#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <cstdint>
#include <vector>

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

#include "Vincia/Vincia.h"
#include "Dire/Dire.h"

using namespace Pythia8;

#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"

#include "ReweightUserHooks.h"
#include "GeneratorInterface/Pythia8Interface/interface/CustomHook.h"
#include "TopRecoilHook.h"

// PS matchning prototype
//
#include "GeneratorInterface/Pythia8Interface/plugins/JetMatchingHook.h"
#include "Pythia8Plugins/JetMatching.h"
#include "Pythia8Plugins/aMCatNLOHooks.h"

#include "GeneratorInterface/Pythia8Interface/interface/MultiUserHook.h"

// Emission Veto Hooks
//
#include "Pythia8Plugins/PowhegHooks.h"
#include "GeneratorInterface/Pythia8Interface/plugins/EmissionVetoHook1.h"

// Resonance scale hook
#include "GeneratorInterface/Pythia8Interface/plugins/PowhegResHook.h"
#include "GeneratorInterface/Pythia8Interface/plugins/PowhegHooksBB4L.h"

//biased tau decayer
#include "GeneratorInterface/Pythia8Interface/interface/BiasedTauDecayer.h"

//decay filter hook
#include "GeneratorInterface/Pythia8Interface/interface/ResonanceDecayFilterHook.h"

//decay filter hook
#include "GeneratorInterface/Pythia8Interface/interface/PTFilterHook.h"

// EvtGen plugin
//
#include "Pythia8Plugins/EvtGen.h"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"
#include "GeneratorInterface/Core/interface/ConcurrentGeneratorFilter.h"
#include "GeneratorInterface/Core/interface/ConcurrentHadronizerFilter.h"

#include "GeneratorInterface/Pythia8Interface/plugins/LHAupLesHouches.h"

#include "HepPID/ParticleIDTranslations.hh"

#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"
#include "GeneratorInterface/ExternalDecays/interface/ConcurrentExternalDecayDriver.h"

namespace CLHEP {
  class HepRandomEngine;
}

using namespace gen;

class Pythia8Hadronizer : public Py8InterfaceBase {
public:
  Pythia8Hadronizer(const edm::ParameterSet &params);
  ~Pythia8Hadronizer() override;

  bool initializeForInternalPartons() override;
  bool initializeForExternalPartons();

  bool generatePartonsAndHadronize() override;
  bool hadronize();

  virtual bool residualDecay();

  void finalizeEvent() override;

  void statistics() override;

  const char *classname() const override { return "Pythia8Hadronizer"; }

  std::unique_ptr<GenLumiInfoHeader> getGenLumiInfoHeader() const override;

private:
  std::unique_ptr<Vincia::VinciaPlugin> fvincia;
  std::unique_ptr<Pythia8::Dire> fDire;

  void doSetRandomEngine(CLHEP::HepRandomEngine *v) override { p8SetRandomEngine(v); }
  std::vector<std::string> const &doSharedResources() const override { return p8SharedResources; }

  /// Center-of-Mass energy
  double comEnergy;

  std::string LHEInputFileName;
  std::unique_ptr<LHAupLesHouches> lhaUP;

  enum { PP, PPbar, ElectronPositron };
  int fInitialState;  // pp, ppbar, or e-e+

  double fBeam1PZ;
  double fBeam2PZ;

  //helper class to allow multiple user hooks simultaneously
  std::unique_ptr<MultiUserHook> fMultiUserHook;

  // Reweight user hooks
  //
  std::unique_ptr<UserHooks> fReweightUserHook;
  std::unique_ptr<UserHooks> fReweightEmpUserHook;
  std::unique_ptr<UserHooks> fReweightRapUserHook;
  std::unique_ptr<UserHooks> fReweightPtHatRapUserHook;

  // PS matching prototype
  //
  std::unique_ptr<JetMatchingHook> fJetMatchingHook;
  std::unique_ptr<Pythia8::JetMatchingMadgraph> fJetMatchingPy8InternalHook;
  std::unique_ptr<Pythia8::amcnlo_unitarised_interface> fMergingHook;

  // Emission Veto Hooks
  //
  std::unique_ptr<PowhegHooks> fEmissionVetoHook;
  std::unique_ptr<EmissionVetoHook1> fEmissionVetoHook1;

  // Resonance scale hook
  std::unique_ptr<PowhegResHook> fPowhegResHook;
  std::unique_ptr<PowhegHooksBB4L> fPowhegHooksBB4L;

  // biased tau decayer
  std::unique_ptr<BiasedTauDecayer> fBiasedTauDecayer;

  //resonance decay filter hook
  std::unique_ptr<ResonanceDecayFilterHook> fResonanceDecayFilterHook;

  //PT filter hook
  std::unique_ptr<PTFilterHook> fPTFilterHook;

  //Generic customized hooks vector
  std::unique_ptr<MultiUserHook> fCustomHooksVector;

  //RecoilToTop userhook
  std::shared_ptr<TopRecoilHook> fTopRecoilHook;

  int EV1_nFinal;
  bool EV1_vetoOn;
  int EV1_maxVetoCount;
  int EV1_pThardMode;
  int EV1_pTempMode;
  int EV1_emittedMode;
  int EV1_pTdefMode;
  bool EV1_MPIvetoOn;
  int EV1_QEDvetoMode;
  int EV1_nFinalMode;

  static const std::vector<std::string> p8SharedResources;

  vector<float> DJR;
  int nME;
  int nMEFiltered;

  int nISRveto;
  int nFSRveto;
};

const std::vector<std::string> Pythia8Hadronizer::p8SharedResources = {edm::SharedResourceNames::kPythia8};

Pythia8Hadronizer::Pythia8Hadronizer(const edm::ParameterSet &params)
    : Py8InterfaceBase(params),
      comEnergy(params.getParameter<double>("comEnergy")),
      LHEInputFileName(params.getUntrackedParameter<std::string>("LHEInputFileName", "")),
      fInitialState(PP),
      nME(-1),
      nMEFiltered(-1),
      nISRveto(0),
      nFSRveto(0) {
  // J.Y.: the following 3 parameters are hacked "for a reason"
  //
  if (params.exists("PPbarInitialState")) {
    if (fInitialState == PP) {
      fInitialState = PPbar;
      edm::LogImportant("GeneratorInterface|Pythia8Interface")
          << "Pythia8 will be initialized for PROTON-ANTIPROTON INITIAL STATE. "
          << "This is a user-request change from the DEFAULT PROTON-PROTON initial state.";
    } else {
      // probably need to throw on attempt to override ?
    }
  } else if (params.exists("ElectronPositronInitialState")) {
    if (fInitialState == PP) {
      fInitialState = ElectronPositron;
      edm::LogInfo("GeneratorInterface|Pythia8Interface")
          << "Pythia8 will be initialized for ELECTRON-POSITRON INITIAL STATE. "
          << "This is a user-request change from the DEFAULT PROTON-PROTON initial state.";
    } else {
      // probably need to throw on attempt to override ?
    }
  } else if (params.exists("ElectronProtonInitialState") || params.exists("PositronProtonInitialState")) {
    // throw on unknown initial state !
    throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
        << " UNKNOWN INITIAL STATE. \n The allowed initial states are: PP, PPbar, ElectronPositron \n";
  }

  // Reweight user hook
  //
  if (params.exists("reweightGen")) {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGen";
    edm::ParameterSet rgParams = params.getParameter<edm::ParameterSet>("reweightGen");
    fReweightUserHook.reset(
        new PtHatReweightUserHook(rgParams.getParameter<double>("pTRef"), rgParams.getParameter<double>("power")));
    edm::LogInfo("Pythia8Interface") << "End setup for reweightGen";
  }
  if (params.exists("reweightGenEmp")) {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGenEmp";
    edm::ParameterSet rgeParams = params.getParameter<edm::ParameterSet>("reweightGenEmp");

    std::string tuneName = "";
    if (rgeParams.exists("tune"))
      tuneName = rgeParams.getParameter<std::string>("tune");
    fReweightEmpUserHook.reset(new PtHatEmpReweightUserHook(tuneName));
    edm::LogInfo("Pythia8Interface") << "End setup for reweightGenEmp";
  }
  if (params.exists("reweightGenRap")) {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGenRap";
    edm::ParameterSet rgrParams = params.getParameter<edm::ParameterSet>("reweightGenRap");
    fReweightRapUserHook.reset(new RapReweightUserHook(rgrParams.getParameter<std::string>("yLabSigmaFunc"),
                                                       rgrParams.getParameter<double>("yLabPower"),
                                                       rgrParams.getParameter<std::string>("yCMSigmaFunc"),
                                                       rgrParams.getParameter<double>("yCMPower"),
                                                       rgrParams.getParameter<double>("pTHatMin"),
                                                       rgrParams.getParameter<double>("pTHatMax")));
    edm::LogInfo("Pythia8Interface") << "End setup for reweightGenRap";
  }
  if (params.exists("reweightGenPtHatRap")) {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGenPtHatRap";
    edm::ParameterSet rgrParams = params.getParameter<edm::ParameterSet>("reweightGenPtHatRap");
    fReweightPtHatRapUserHook.reset(new PtHatRapReweightUserHook(rgrParams.getParameter<std::string>("yLabSigmaFunc"),
                                                                 rgrParams.getParameter<double>("yLabPower"),
                                                                 rgrParams.getParameter<std::string>("yCMSigmaFunc"),
                                                                 rgrParams.getParameter<double>("yCMPower"),
                                                                 rgrParams.getParameter<double>("pTHatMin"),
                                                                 rgrParams.getParameter<double>("pTHatMax")));
    edm::LogInfo("Pythia8Interface") << "End setup for reweightGenPtHatRap";
  }

  if (params.exists("useUserHook"))
    throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
        << " Obsolete parameter: useUserHook \n Please use the actual one instead \n";

  // PS matching prototype
  //
  if (params.exists("jetMatching")) {
    edm::ParameterSet jmParams = params.getUntrackedParameter<edm::ParameterSet>("jetMatching");
    std::string scheme = jmParams.getParameter<std::string>("scheme");
    if (scheme == "Madgraph" || scheme == "MadgraphFastJet") {
      fJetMatchingHook.reset(new JetMatchingHook(jmParams, &fMasterGen->info));
    }
  }

  // Pythia8Interface emission veto
  //
  if (params.exists("emissionVeto1")) {
    EV1_nFinal = -1;
    if (params.exists("EV1_nFinal"))
      EV1_nFinal = params.getParameter<int>("EV1_nFinal");
    EV1_vetoOn = true;
    if (params.exists("EV1_vetoOn"))
      EV1_vetoOn = params.getParameter<bool>("EV1_vetoOn");
    EV1_maxVetoCount = 10;
    if (params.exists("EV1_maxVetoCount"))
      EV1_maxVetoCount = params.getParameter<int>("EV1_maxVetoCount");
    EV1_pThardMode = 1;
    if (params.exists("EV1_pThardMode"))
      EV1_pThardMode = params.getParameter<int>("EV1_pThardMode");
    EV1_pTempMode = 0;
    if (params.exists("EV1_pTempMode"))
      EV1_pTempMode = params.getParameter<int>("EV1_pTempMode");
    if (EV1_pTempMode > 2 || EV1_pTempMode < 0)
      throw edm::Exception(edm::errors::Configuration, "Pythia8Interface") << " Wrong value for EV1_pTempMode code\n";
    EV1_emittedMode = 0;
    if (params.exists("EV1_emittedMode"))
      EV1_emittedMode = params.getParameter<int>("EV1_emittedMode");
    EV1_pTdefMode = 1;
    if (params.exists("EV1_pTdefMode"))
      EV1_pTdefMode = params.getParameter<int>("EV1_pTdefMode");
    EV1_MPIvetoOn = false;
    if (params.exists("EV1_MPIvetoOn"))
      EV1_MPIvetoOn = params.getParameter<bool>("EV1_MPIvetoOn");
    EV1_QEDvetoMode = 0;
    if (params.exists("EV1_QEDvetoMode"))
      EV1_QEDvetoMode = params.getParameter<int>("EV1_QEDvetoMode");
    EV1_nFinalMode = 0;
    if (params.exists("EV1_nFinalMode"))
      EV1_nFinalMode = params.getParameter<int>("EV1_nFinalMode");
    fEmissionVetoHook1.reset(new EmissionVetoHook1(EV1_nFinal,
                                                   EV1_vetoOn,
                                                   EV1_maxVetoCount,
                                                   EV1_pThardMode,
                                                   EV1_pTempMode,
                                                   EV1_emittedMode,
                                                   EV1_pTdefMode,
                                                   EV1_MPIvetoOn,
                                                   EV1_QEDvetoMode,
                                                   EV1_nFinalMode,
                                                   0));
  }

  fCustomHooksVector.reset(new MultiUserHook);
  if (params.exists("UserCustomization")) {
    const std::vector<edm::ParameterSet> userParams =
        params.getParameter<std::vector<edm::ParameterSet>>("UserCustomization");
    for (const auto &pluginParams : userParams) {
      fCustomHooksVector->addHook(
          CustomHookFactory::get()->create(pluginParams.getParameter<std::string>("pluginName"), pluginParams));
    }
  }

  if (params.exists("VinciaPlugin")) {
    fMasterGen.reset(new Pythia);
    fvincia.reset(new Vincia::VinciaPlugin(fMasterGen.get()));
  }
  if (params.exists("DirePlugin")) {
    fMasterGen.reset(new Pythia);
    fDire.reset(new Pythia8::Dire());
    fDire->initSettings(*fMasterGen.get());
    fDire->initShowersAndWeights(*fMasterGen.get(), nullptr, nullptr);
  }
}

Pythia8Hadronizer::~Pythia8Hadronizer() {}

bool Pythia8Hadronizer::initializeForInternalPartons() {
  bool status = false, status1 = false;

  if (lheFile_.empty()) {
    if (fInitialState == PP)  // default
    {
      fMasterGen->settings.mode("Beams:idA", 2212);
      fMasterGen->settings.mode("Beams:idB", 2212);
    } else if (fInitialState == PPbar) {
      fMasterGen->settings.mode("Beams:idA", 2212);
      fMasterGen->settings.mode("Beams:idB", -2212);
    } else if (fInitialState == ElectronPositron) {
      fMasterGen->settings.mode("Beams:idA", 11);
      fMasterGen->settings.mode("Beams:idB", -11);
    } else {
      // throw on unknown initial state !
      throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
          << " UNKNOWN INITIAL STATE. \n The allowed initial states are: PP, PPbar, ElectronPositron \n";
    }
    fMasterGen->settings.parm("Beams:eCM", comEnergy);
  } else {
    fMasterGen->settings.mode("Beams:frameType", 4);
    fMasterGen->settings.word("Beams:LHEF", lheFile_);
  }

  fMultiUserHook.reset(new MultiUserHook);

  if (fReweightUserHook.get())
    fMultiUserHook->addHook(fReweightUserHook.get());
  if (fReweightEmpUserHook.get())
    fMultiUserHook->addHook(fReweightEmpUserHook.get());
  if (fReweightRapUserHook.get())
    fMultiUserHook->addHook(fReweightRapUserHook.get());
  if (fReweightPtHatRapUserHook.get())
    fMultiUserHook->addHook(fReweightPtHatRapUserHook.get());
  if (fJetMatchingHook.get())
    fMultiUserHook->addHook(fJetMatchingHook.get());
  if (fEmissionVetoHook1.get()) {
    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook 1 from CMSSW Pythia8Interface";
    fMultiUserHook->addHook(fEmissionVetoHook1.get());
  }

  if (fMasterGen->settings.mode("POWHEG:veto") > 0 || fMasterGen->settings.mode("POWHEG:MPIveto") > 0) {
    if (fJetMatchingHook.get() || fEmissionVetoHook1.get())
      throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
          << " Attempt to turn on PowhegHooks by pythia8 settings but there are incompatible hooks on \n Incompatible "
             "are : jetMatching, emissionVeto1 \n";

    fEmissionVetoHook.reset(new PowhegHooks());

    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook from pythia8 code";
    fMultiUserHook->addHook(fEmissionVetoHook.get());
  }

  bool PowhegRes = fMasterGen->settings.flag("POWHEGres:calcScales");
  if (PowhegRes) {
    edm::LogInfo("Pythia8Interface") << "Turning on resonance scale setting from CMSSW Pythia8Interface";
    fPowhegResHook.reset(new PowhegResHook());
    fMultiUserHook->addHook(fPowhegResHook.get());
  }

  bool PowhegBB4L = fMasterGen->settings.flag("POWHEG:bb4l");
  if (PowhegBB4L) {
    edm::LogInfo("Pythia8Interface") << "Turning on BB4l hook from CMSSW Pythia8Interface";
    fPowhegHooksBB4L.reset(new PowhegHooksBB4L());
    fMultiUserHook->addHook(fPowhegHooksBB4L.get());
  }

  bool TopRecoilHook1 = fMasterGen->settings.flag("TopRecoilHook:doTopRecoilIn");
  if (TopRecoilHook1) {
    edm::LogInfo("Pythia8Interface") << "Turning on RecoilToTop hook from Pythia8Interface";
    fTopRecoilHook.reset(new TopRecoilHook());
    fMultiUserHook->addHook(fTopRecoilHook.get());
  }

  //adapted from main89.cc in pythia8 examples
  bool internalMatching = fMasterGen->settings.flag("JetMatching:merge");
  bool internalMerging = !(fMasterGen->settings.word("Merging:Process") == "void");

  if (internalMatching && internalMerging) {
    throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
        << " Only one jet matching/merging scheme can be used at a time. \n";
  }

  if (internalMatching) {
    fJetMatchingPy8InternalHook.reset(new Pythia8::JetMatchingMadgraph);
    fMultiUserHook->addHook(fJetMatchingPy8InternalHook.get());
  }

  if (internalMerging) {
    int scheme = (fMasterGen->settings.flag("Merging:doUMEPSTree") || fMasterGen->settings.flag("Merging:doUMEPSSubt"))
                     ? 1
                     : ((fMasterGen->settings.flag("Merging:doUNLOPSTree") ||
                         fMasterGen->settings.flag("Merging:doUNLOPSSubt") ||
                         fMasterGen->settings.flag("Merging:doUNLOPSLoop") ||
                         fMasterGen->settings.flag("Merging:doUNLOPSSubtNLO"))
                            ? 2
                            : 0);
    fMergingHook.reset(new Pythia8::amcnlo_unitarised_interface(scheme));
    fMultiUserHook->addHook(fMergingHook.get());
  }

  bool biasedTauDecayer = fMasterGen->settings.flag("BiasedTauDecayer:filter");
  if (biasedTauDecayer) {
    fBiasedTauDecayer.reset(new BiasedTauDecayer(&(fMasterGen->info),
                                                 &(fMasterGen->settings),
                                                 &(fMasterGen->particleData),
                                                 &(fMasterGen->rndm),
                                                 &(fMasterGen->couplings)));
    std::vector<int> handledParticles;
    handledParticles.push_back(15);
    fMasterGen->setDecayPtr(fBiasedTauDecayer.get(), handledParticles);
  }

  bool resonanceDecayFilter = fMasterGen->settings.flag("ResonanceDecayFilter:filter");
  if (resonanceDecayFilter) {
    fResonanceDecayFilterHook.reset(new ResonanceDecayFilterHook);
    fMultiUserHook->addHook(fResonanceDecayFilterHook.get());
  }

  bool PTFilter = fMasterGen->settings.flag("PTFilter:filter");
  if (PTFilter) {
    fPTFilterHook.reset(new PTFilterHook);
    fMultiUserHook->addHook(fPTFilterHook.get());
  }

  if (fCustomHooksVector->nHooks() > 0) {
    edm::LogInfo("Pythia8Interface") << "Adding customized user hooks";
    for (const auto &fUserHook : fCustomHooksVector.get()->hooks()) {
      fMultiUserHook->addHook(fUserHook);
    }
  }

  if (fMultiUserHook->nHooks() > 0) {
    fMasterGen->setUserHooksPtr(fMultiUserHook.get());
  }

  edm::LogInfo("Pythia8Interface") << "Initializing MasterGen";
  if (fvincia.get()) {
    fvincia->init();
    status = true;
  } else if (fDire.get()) {
    //fDire->initTune(*fMasterGen.get());
    fDire->weightsPtr->setup();
    fMasterGen->init();
    fDire->setup(*fMasterGen.get());
    status = true;
  } else {
    status = fMasterGen->init();
  }

  //clean up temp file
  if (!slhafile_.empty()) {
    std::remove(slhafile_.c_str());
  }

  if (pythiaPylistVerbosity > 10) {
    if (pythiaPylistVerbosity == 11 || pythiaPylistVerbosity == 13)
      fMasterGen->settings.listAll();
    if (pythiaPylistVerbosity == 12 || pythiaPylistVerbosity == 13)
      fMasterGen->particleData.listAll();
  }

  // init decayer
  fDecayer->settings.flag("ProcessLevel:all", false);  // trick
  fDecayer->settings.flag("ProcessLevel:resonanceDecays", true);
  edm::LogInfo("Pythia8Interface") << "Initializing Decayer";
  status1 = fDecayer->init();

  if (useEvtGen) {
    edm::LogInfo("Pythia8Hadronizer") << "Creating and initializing pythia8 EvtGen plugin";
    evtgenDecays.reset(new EvtGenDecays(fMasterGen.get(), evtgenDecFile, evtgenPdlFile));
    for (unsigned int i = 0; i < evtgenUserFiles.size(); i++)
      evtgenDecays->readDecayFile(evtgenUserFiles.at(i));
  }

  return (status && status1);
}

bool Pythia8Hadronizer::initializeForExternalPartons() {
  edm::LogInfo("Pythia8Interface") << "Initializing for external partons";

  bool status = false, status1 = false;

  fMultiUserHook.reset(new MultiUserHook);

  if (fReweightUserHook.get())
    fMultiUserHook->addHook(fReweightUserHook.get());
  if (fReweightEmpUserHook.get())
    fMultiUserHook->addHook(fReweightEmpUserHook.get());
  if (fReweightRapUserHook.get())
    fMultiUserHook->addHook(fReweightRapUserHook.get());
  if (fReweightPtHatRapUserHook.get())
    fMultiUserHook->addHook(fReweightPtHatRapUserHook.get());
  if (fJetMatchingHook.get())
    fMultiUserHook->addHook(fJetMatchingHook.get());
  if (fEmissionVetoHook1.get()) {
    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook 1 from CMSSW Pythia8Interface";
    fMultiUserHook->addHook(fEmissionVetoHook1.get());
  }

  if (fCustomHooksVector->nHooks() > 0) {
    edm::LogInfo("Pythia8Interface") << "Adding customized user hooks";
    for (const auto &fUserHook : fCustomHooksVector.get()->hooks()) {
      fMultiUserHook->addHook(fUserHook);
    }
  }

  if (fMasterGen->settings.mode("POWHEG:veto") > 0 || fMasterGen->settings.mode("POWHEG:MPIveto") > 0) {
    if (fJetMatchingHook.get() || fEmissionVetoHook1.get())
      throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
          << " Attempt to turn on PowhegHooks by pythia8 settings but there are incompatible hooks on \n Incompatible "
             "are : jetMatching, emissionVeto1 \n";

    fEmissionVetoHook.reset(new PowhegHooks());

    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook from pythia8 code";
    fMultiUserHook->addHook(fEmissionVetoHook.get());
  }

  bool PowhegRes = fMasterGen->settings.flag("POWHEGres:calcScales");
  if (PowhegRes) {
    edm::LogInfo("Pythia8Interface") << "Turning on resonance scale setting from CMSSW Pythia8Interface";
    fPowhegResHook.reset(new PowhegResHook());
    fMultiUserHook->addHook(fPowhegResHook.get());
  }

  bool PowhegBB4L = fMasterGen->settings.flag("POWHEG:bb4l");
  if (PowhegBB4L) {
    edm::LogInfo("Pythia8Interface") << "Turning on BB4l hook from CMSSW Pythia8Interface";
    fPowhegHooksBB4L.reset(new PowhegHooksBB4L());
    fMultiUserHook->addHook(fPowhegHooksBB4L.get());
  }

  bool TopRecoilHook1 = fMasterGen->settings.flag("TopRecoilHook:doTopRecoilIn");
  if (TopRecoilHook1) {
    edm::LogInfo("Pythia8Interface") << "Turning on RecoilToTop hook from Pythia8Interface";
    fTopRecoilHook.reset(new TopRecoilHook());
    fMultiUserHook->addHook(fTopRecoilHook.get());
  }

  //adapted from main89.cc in pythia8 examples
  bool internalMatching = fMasterGen->settings.flag("JetMatching:merge");
  bool internalMerging = !(fMasterGen->settings.word("Merging:Process") == "void");

  if (internalMatching && internalMerging) {
    throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
        << " Only one jet matching/merging scheme can be used at a time. \n";
  }

  if (internalMatching) {
    fJetMatchingPy8InternalHook.reset(new Pythia8::JetMatchingMadgraph);
    fMultiUserHook->addHook(fJetMatchingPy8InternalHook.get());
  }

  if (internalMerging) {
    int scheme = (fMasterGen->settings.flag("Merging:doUMEPSTree") || fMasterGen->settings.flag("Merging:doUMEPSSubt"))
                     ? 1
                     : ((fMasterGen->settings.flag("Merging:doUNLOPSTree") ||
                         fMasterGen->settings.flag("Merging:doUNLOPSSubt") ||
                         fMasterGen->settings.flag("Merging:doUNLOPSLoop") ||
                         fMasterGen->settings.flag("Merging:doUNLOPSSubtNLO"))
                            ? 2
                            : 0);
    fMergingHook.reset(new Pythia8::amcnlo_unitarised_interface(scheme));
    fMultiUserHook->addHook(fMergingHook.get());
  }

  bool biasedTauDecayer = fMasterGen->settings.flag("BiasedTauDecayer:filter");
  if (biasedTauDecayer) {
    fBiasedTauDecayer.reset(new BiasedTauDecayer(&(fMasterGen->info),
                                                 &(fMasterGen->settings),
                                                 &(fMasterGen->particleData),
                                                 &(fMasterGen->rndm),
                                                 &(fMasterGen->couplings)));
    std::vector<int> handledParticles;
    handledParticles.push_back(15);
    fMasterGen->setDecayPtr(fBiasedTauDecayer.get(), handledParticles);
  }

  bool resonanceDecayFilter = fMasterGen->settings.flag("ResonanceDecayFilter:filter");
  if (resonanceDecayFilter) {
    fResonanceDecayFilterHook.reset(new ResonanceDecayFilterHook);
    fMultiUserHook->addHook(fResonanceDecayFilterHook.get());
  }

  bool PTFilter = fMasterGen->settings.flag("PTFilter:filter");
  if (PTFilter) {
    fPTFilterHook.reset(new PTFilterHook);
    fMultiUserHook->addHook(fPTFilterHook.get());
  }

  if (fMultiUserHook->nHooks() > 0) {
    fMasterGen->setUserHooksPtr(fMultiUserHook.get());
  }

  if (!LHEInputFileName.empty()) {
    edm::LogInfo("Pythia8Interface") << "Initialize direct pythia8 reading from LHE file " << LHEInputFileName;
    edm::LogInfo("Pythia8Interface") << "Some LHE information can be not stored";
    fMasterGen->settings.mode("Beams:frameType", 4);
    fMasterGen->settings.word("Beams:LHEF", LHEInputFileName);
    status = fMasterGen->init();

  } else {
    lhaUP.reset(new LHAupLesHouches());
    lhaUP->setScalesFromLHEF(fMasterGen->settings.flag("Beams:setProductionScalesFromLHEF"));
    lhaUP->loadRunInfo(lheRunInfo());

    if (fJetMatchingHook.get()) {
      fJetMatchingHook->init(lheRunInfo());
    }

    fMasterGen->settings.mode("Beams:frameType", 5);
    fMasterGen->setLHAupPtr(lhaUP.get());
    edm::LogInfo("Pythia8Interface") << "Initializing MasterGen";
    status = fMasterGen->init();
  }

  //clean up temp file
  if (!slhafile_.empty()) {
    std::remove(slhafile_.c_str());
  }

  if (pythiaPylistVerbosity > 10) {
    if (pythiaPylistVerbosity == 11 || pythiaPylistVerbosity == 13)
      fMasterGen->settings.listAll();
    if (pythiaPylistVerbosity == 12 || pythiaPylistVerbosity == 13)
      fMasterGen->particleData.listAll();
  }

  // init decayer
  fDecayer->settings.flag("ProcessLevel:all", false);  // trick
  fDecayer->settings.flag("ProcessLevel:resonanceDecays", true);
  edm::LogInfo("Pythia8Interface") << "Initializing Decayer";
  status1 = fDecayer->init();

  if (useEvtGen) {
    edm::LogInfo("Pythia8Hadronizer") << "Creating and initializing pythia8 EvtGen plugin";
    evtgenDecays.reset(new EvtGenDecays(fMasterGen.get(), evtgenDecFile, evtgenPdlFile));
    for (unsigned int i = 0; i < evtgenUserFiles.size(); i++)
      evtgenDecays->readDecayFile(evtgenUserFiles.at(i));
  }

  return (status && status1);
}

void Pythia8Hadronizer::statistics() {
  fMasterGen->stat();

  if (fEmissionVetoHook.get()) {
    edm::LogPrint("Pythia8Interface") << "\n"
                                      << "Number of ISR vetoed = " << nISRveto;
    edm::LogPrint("Pythia8Interface") << "Number of FSR vetoed = " << nFSRveto;
  }

  double xsec = fMasterGen->info.sigmaGen();  // cross section in mb
  xsec *= 1.0e9;                              // translate to pb (CMS/Gen "convention" as of May 2009)
  double err = fMasterGen->info.sigmaErr();   // cross section err in mb
  err *= 1.0e9;                               // translate to pb (CMS/Gen "convention" as of May 2009)
  runInfo().setInternalXSec(GenRunInfoProduct::XSec(xsec, err));
}

bool Pythia8Hadronizer::generatePartonsAndHadronize() {
  DJR.resize(0);
  nME = -1;
  nMEFiltered = -1;

  if (fJetMatchingHook.get()) {
    fJetMatchingHook->resetMatchingStatus();
    fJetMatchingHook->beforeHadronization(lheEvent());
  }

  if (!fMasterGen->next())
    return false;

  double mergeweight = fMasterGen.get()->info.mergingWeightNLO();
  if (fMergingHook.get()) {
    mergeweight *= fMergingHook->getNormFactor();
  }

  //protect against 0-weight from ckkw or similar
  if (std::abs(mergeweight) == 0.) {
    event().reset();
    return false;
  }

  if (fJetMatchingPy8InternalHook.get()) {
    const std::vector<double> djrmatch = fJetMatchingPy8InternalHook->getDJR();
    //cap size of djr vector to save storage space (keep only up to first 6 elements)
    unsigned int ndjr = std::min(djrmatch.size(), std::vector<double>::size_type(6));
    for (unsigned int idjr = 0; idjr < ndjr; ++idjr) {
      DJR.push_back(djrmatch[idjr]);
    }

    nME = fJetMatchingPy8InternalHook->nMEpartons().first;
    nMEFiltered = fJetMatchingPy8InternalHook->nMEpartons().second;
  }

  if (evtgenDecays.get())
    evtgenDecays->decay();

  event().reset(new HepMC::GenEvent);
  bool py8hepmc = toHepMC.fill_next_event(*(fMasterGen.get()), event().get());

  if (!py8hepmc) {
    return false;
  }

  //add ckkw/umeps/unlops merging weight
  if (mergeweight != 1.) {
    event()->weights()[0] *= mergeweight;
  }

  if (fEmissionVetoHook.get()) {
    nISRveto += fEmissionVetoHook->getNISRveto();
    nFSRveto += fEmissionVetoHook->getNFSRveto();
  }

  //fill additional weights for systematic uncertainties
  if (fMasterGen->info.getWeightsDetailedSize() > 0) {
    for (const string &key : fMasterGen->info.initrwgt->weightsKeys) {
      double wgt = (*fMasterGen->info.weights_detailed)[key];
      event()->weights().push_back(wgt);
    }
  } else if (fMasterGen->info.getWeightsCompressedSize() > 0) {
    for (unsigned int i = 0; i < fMasterGen->info.getWeightsCompressedSize(); i++) {
      double wgt = fMasterGen->info.getWeightsCompressedValue(i);
      event()->weights().push_back(wgt);
    }
  }

  // fill shower weights
  // http://home.thep.lu.se/~torbjorn/pythia82html/Variations.html
  if (fMasterGen->info.nWeights() > 1) {
    for (int i = 0; i < fMasterGen->info.nWeights(); ++i) {
      double wgt = fMasterGen->info.weight(i);
      event()->weights().push_back(wgt);
    }
  }

  // VINCIA shower weights
  // http://vincia.hepforge.org/current/share/Vincia/htmldoc/VinciaUncertainties.html
  if (fvincia.get()) {
    event()->weights()[0] *= fvincia->weight(0);
    for (int iVar = 1; iVar < fvincia->nWeights(); iVar++) {
      event()->weights().push_back(fvincia->weight(iVar));
    }
  }

  // Retrieve Dire shower weights
  if (fDire.get()) {
    fDire->weightsPtr->calcWeight(0.);
    fDire->weightsPtr->reset();

    //Make sure the base weight comes first
    event()->weights()[0] *= fDire->weightsPtr->getShowerWeight("base");

    map<string, double>::iterator it;
    for (it = fDire->weightsPtr->getShowerWeights()->begin(); it != fDire->weightsPtr->getShowerWeights()->end();
         it++) {
      if (it->first == "base")
        continue;
      event()->weights().push_back(it->second);
    }
  }

  return true;
}

bool Pythia8Hadronizer::hadronize() {
  DJR.resize(0);
  nME = -1;
  nMEFiltered = -1;
  if (LHEInputFileName.empty())
    lhaUP->loadEvent(lheEvent());

  if (fJetMatchingHook.get()) {
    fJetMatchingHook->resetMatchingStatus();
    fJetMatchingHook->beforeHadronization(lheEvent());
  }

  bool py8next = fMasterGen->next();

  double mergeweight = fMasterGen.get()->info.mergingWeightNLO();
  if (fMergingHook.get()) {
    mergeweight *= fMergingHook->getNormFactor();
  }

  //protect against 0-weight from ckkw or similar
  if (!py8next || std::abs(mergeweight) == 0.) {
    lheEvent()->count(lhef::LHERunInfo::kSelected, 1.0, mergeweight);
    event().reset();
    return false;
  }

  if (fJetMatchingPy8InternalHook.get()) {
    const std::vector<double> djrmatch = fJetMatchingPy8InternalHook->getDJR();
    //cap size of djr vector to save storage space (keep only up to first 6 elements)
    unsigned int ndjr = std::min(djrmatch.size(), std::vector<double>::size_type(6));
    for (unsigned int idjr = 0; idjr < ndjr; ++idjr) {
      DJR.push_back(djrmatch[idjr]);
    }

    nME = fJetMatchingPy8InternalHook->nMEpartons().first;
    nMEFiltered = fJetMatchingPy8InternalHook->nMEpartons().second;
  }

  // update LHE matching statistics
  //
  lheEvent()->count(lhef::LHERunInfo::kAccepted, 1.0, mergeweight);

  if (evtgenDecays.get())
    evtgenDecays->decay();

  event().reset(new HepMC::GenEvent);
  bool py8hepmc = toHepMC.fill_next_event(*(fMasterGen.get()), event().get());
  if (!py8hepmc) {
    return false;
  }

  //add ckkw/umeps/unlops merging weight
  if (mergeweight != 1.) {
    event()->weights()[0] *= mergeweight;
  }

  if (fEmissionVetoHook.get()) {
    nISRveto += fEmissionVetoHook->getNISRveto();
    nFSRveto += fEmissionVetoHook->getNFSRveto();
  }

  // fill shower weights
  // http://home.thep.lu.se/~torbjorn/pythia82html/Variations.html
  if (fMasterGen->info.nWeights() > 1) {
    for (int i = 0; i < fMasterGen->info.nWeights(); ++i) {
      double wgt = fMasterGen->info.weight(i);
      event()->weights().push_back(wgt);
    }
  }

  return true;
}

bool Pythia8Hadronizer::residualDecay() {
  Event *pythiaEvent = &(fMasterGen->event);

  int NPartsBeforeDecays = pythiaEvent->size();
  int NPartsAfterDecays = event().get()->particles_size();

  if (NPartsAfterDecays == NPartsBeforeDecays)
    return true;

  bool result = true;

  for (int ipart = NPartsAfterDecays; ipart > NPartsBeforeDecays; ipart--) {
    HepMC::GenParticle *part = event().get()->barcode_to_particle(ipart);

    if (part->status() == 1 && (fDecayer->particleData).canDecay(part->pdg_id())) {
      fDecayer->event.reset();
      Particle py8part(part->pdg_id(),
                       93,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       part->momentum().x(),
                       part->momentum().y(),
                       part->momentum().z(),
                       part->momentum().t(),
                       part->generated_mass());
      HepMC::GenVertex *ProdVtx = part->production_vertex();
      py8part.vProd(ProdVtx->position().x(), ProdVtx->position().y(), ProdVtx->position().z(), ProdVtx->position().t());
      py8part.tau((fDecayer->particleData).tau0(part->pdg_id()));
      fDecayer->event.append(py8part);
      int nentries = fDecayer->event.size();
      if (!fDecayer->event[nentries - 1].mayDecay())
        continue;
      fDecayer->next();
      int nentries1 = fDecayer->event.size();
      if (nentries1 <= nentries)
        continue;  //same number of particles, no decays...

      part->set_status(2);

      result = toHepMC.fill_next_event(*(fDecayer.get()), event().get(), -1, true, part);
    }
  }

  return result;
}

void Pythia8Hadronizer::finalizeEvent() {
  bool lhe = lheEvent() != nullptr;

  // now create the GenEventInfo product from the GenEvent and fill
  // the missing pieces
  eventInfo().reset(new GenEventInfoProduct(event().get()));

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

  if (maxEventsToPrint > 0 && (pythiaPylistVerbosity || pythiaHepMCVerbosity || pythiaHepMCVerbosityParticles)) {
    maxEventsToPrint--;
    if (pythiaPylistVerbosity) {
      fMasterGen->info.list();
      fMasterGen->event.list();
    }

    if (pythiaHepMCVerbosity) {
      std::cout << "Event process = " << fMasterGen->info.code() << "\n"
                << "----------------------" << std::endl;
      event()->print();
    }
    if (pythiaHepMCVerbosityParticles) {
      std::cout << "Event process = " << fMasterGen->info.code() << "\n"
                << "----------------------" << std::endl;
      ascii_io->write_event(event().get());
    }
  }
}

std::unique_ptr<GenLumiInfoHeader> Pythia8Hadronizer::getGenLumiInfoHeader() const {
  auto genLumiInfoHeader = BaseHadronizer::getGenLumiInfoHeader();

  //fill lhe headers
  //*FIXME* initrwgt header is corrupt due to pythia bug
  for (const std::string &key : fMasterGen->info.headerKeys()) {
    genLumiInfoHeader->lheHeaders().emplace_back(key, fMasterGen->info.header(key));
  }

  //check, if it is not only nominal weight
  int weights_number = fMasterGen->info.nWeights();
  if (fMasterGen->info.initrwgt)
    weights_number += fMasterGen->info.initrwgt->weightsKeys.size();
  if (weights_number > 1) {
    genLumiInfoHeader->weightNames().reserve(weights_number + 1);
    genLumiInfoHeader->weightNames().push_back("nominal");
  }

  //fill weight names
  if (fMasterGen->info.initrwgt) {
    for (const std::string &key : fMasterGen->info.initrwgt->weightsKeys) {
      std::string weightgroupname;
      for (const auto &wgtgrp : fMasterGen->info.initrwgt->weightgroups) {
        const auto &wgtgrpwgt = wgtgrp.second.weights.find(key);
        if (wgtgrpwgt != wgtgrp.second.weights.end()) {
          weightgroupname = wgtgrp.first;
        }
      }

      std::ostringstream weightname;
      weightname << "LHE, id = " << key << ", ";
      if (!weightgroupname.empty()) {
        weightname << "group = " << weightgroupname << ", ";
      }
      weightname << fMasterGen->info.initrwgt->weights[key].contents;
      genLumiInfoHeader->weightNames().push_back(weightname.str());
    }
  }

  //fill shower labels
  // http://home.thep.lu.se/~torbjorn/pythia82html/Variations.html
  // http://home.thep.lu.se/~torbjorn/doxygen/classPythia8_1_1Info.html
  if (fMasterGen->info.nWeights() > 1) {
    for (int i = 0; i < fMasterGen->info.nWeights(); ++i) {
      genLumiInfoHeader->weightNames().push_back(fMasterGen->info.weightLabel(i));
    }
  }

  // VINCIA shower weights
  // http://vincia.hepforge.org/current/share/Vincia/htmldoc/VinciaUncertainties.html
  if (fvincia.get()) {
    for (int iVar = 0; iVar < fvincia->nWeights(); iVar++) {
      genLumiInfoHeader->weightNames().push_back(fvincia->weightLabel(iVar));
    }
  }

  if (fDire.get()) {
    //Make sure the base weight comes first
    genLumiInfoHeader->weightNames().push_back("base");

    map<string, double>::iterator it;
    for (it = fDire->weightsPtr->getShowerWeights()->begin(); it != fDire->weightsPtr->getShowerWeights()->end();
         it++) {
      if (it->first == "base")
        continue;
      genLumiInfoHeader->weightNames().push_back(it->first);
    }
  }

  return genLumiInfoHeader;
}

typedef edm::GeneratorFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8GeneratorFilter;
DEFINE_FWK_MODULE(Pythia8GeneratorFilter);

typedef edm::HadronizerFilter<Pythia8Hadronizer, ExternalDecayDriver> Pythia8HadronizerFilter;
DEFINE_FWK_MODULE(Pythia8HadronizerFilter);

typedef edm::ConcurrentGeneratorFilter<Pythia8Hadronizer, ConcurrentExternalDecayDriver>
    Pythia8ConcurrentGeneratorFilter;
DEFINE_FWK_MODULE(Pythia8ConcurrentGeneratorFilter);

typedef edm::ConcurrentHadronizerFilter<Pythia8Hadronizer, ConcurrentExternalDecayDriver>
    Pythia8ConcurrentHadronizerFilter;
DEFINE_FWK_MODULE(Pythia8ConcurrentHadronizerFilter);

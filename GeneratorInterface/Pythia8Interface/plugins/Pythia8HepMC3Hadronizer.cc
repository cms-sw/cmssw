#include <iostream>
#include <memory>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/Print.h"

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"

using namespace Pythia8;

#include "GeneratorInterface/Pythia8Interface/interface/Py8HMC3InterfaceBase.h"

#include "ReweightUserHooks.h"
#include "GeneratorInterface/Pythia8Interface/interface/CustomHook.h"
#include "TopRecoilHook.h"

// PS matchning prototype
//
#include "GeneratorInterface/Pythia8Interface/plugins/JetMatchingHook.h"
#include "Pythia8Plugins/JetMatching.h"
#include "Pythia8Plugins/aMCatNLOHooks.h"

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

//Insert class for use w/ PDFPtr for proton-photon flux
//parameters hardcoded according to main70.cc in PYTHIA8 v3.10
class Nucleus2gamma2 : public Pythia8::PDF {
private:
  double radius;
  int z;

public:
  // Constructor.
  Nucleus2gamma2(int idBeamIn, double R = -1.0, int Z = -1) : Pythia8::PDF(idBeamIn), radius(R), z(Z) {}

  void xfUpdate(int, double x, double) override {
    if (z == -1) {
      // lead
      if (idBeam == 1000822080)
        z = 82;
    }
    if (radius == -1) {
      // lead
      if (idBeam == 1000822080)
        radius = 6.636;
    }

    if (z < 0 || radius < 0)
      throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
          << " Invalid photon flux input parameters: beam ID= " << idBeam << " , radius= " << radius << " , z= " << z
          << "\n";

    // Minimum impact parameter (~2*radius) [fm].
    double bmin = 2 * radius;

    // Per-nucleon mass for lead.
    double m2 = pow2(0.9314);
    double alphaEM = 0.007297353080;
    double hbarc = 0.197;
    double xi = x * sqrt(m2) * bmin / hbarc;
    double bK0 = besselK0(xi);
    double bK1 = besselK1(xi);
    double intB = xi * bK1 * bK0 - 0.5 * pow2(xi) * (pow2(bK1) - pow2(bK0));
    xgamma = 2. * alphaEM * pow2(z) / M_PI * intB;
  }
};

class Pythia8HepMC3Hadronizer : public Py8HMC3InterfaceBase {
public:
  Pythia8HepMC3Hadronizer(const edm::ParameterSet &params);
  ~Pythia8HepMC3Hadronizer() override = default;

  bool initializeForInternalPartons() override;
  bool initializeForExternalPartons();

  bool generatePartonsAndHadronize() override;
  bool hadronize();

  virtual bool residualDecay();

  void finalizeEvent() override;

  void statistics() override;

  const char *classname() const override { return "Pythia8HepMC3Hadronizer"; }

  std::unique_ptr<GenLumiInfoHeader> getGenLumiInfoHeader() const override;

private:
  void doSetRandomEngine(CLHEP::HepRandomEngine *v) override { p8SetRandomEngine(v); }
  std::vector<std::string> const &doSharedResources() const override { return p8SharedResources; }

  /// Center-of-Mass energy
  double comEnergy;

  std::string LHEInputFileName;
  std::shared_ptr<LHAupLesHouches> lhaUP;

  enum { PP, PPbar, ElectronPositron };
  int fInitialState;  // pp, ppbar, or e-e+

  double fBeam1PZ;
  double fBeam2PZ;

  //PDFPtr for the photonFlux
  //Following main70.cc example in PYTHIA8 v3.10
  edm::ParameterSet photonFluxParams;

  //helper class to allow multiple user hooks simultaneously
  std::shared_ptr<UserHooksVector> fUserHooksVector;
  bool UserHooksSet;

  // Reweight user hooks
  //
  std::shared_ptr<UserHooks> fReweightUserHook;
  std::shared_ptr<UserHooks> fReweightEmpUserHook;
  std::shared_ptr<UserHooks> fReweightRapUserHook;
  std::shared_ptr<UserHooks> fReweightPtHatRapUserHook;

  // PS matching prototype
  //
  std::shared_ptr<JetMatchingHook> fJetMatchingHook;
  std::shared_ptr<Pythia8::JetMatchingMadgraph> fJetMatchingPy8InternalHook;
  std::shared_ptr<Pythia8::amcnlo_unitarised_interface> fMergingHook;

  // Emission Veto Hooks
  //
  std::shared_ptr<PowhegHooks> fEmissionVetoHook;
  std::shared_ptr<EmissionVetoHook1> fEmissionVetoHook1;

  // Resonance scale hook
  std::shared_ptr<PowhegResHook> fPowhegResHook;
  std::shared_ptr<PowhegHooksBB4L> fPowhegHooksBB4L;

  // biased tau decayer
  std::shared_ptr<BiasedTauDecayer> fBiasedTauDecayer;

  //resonance decay filter hook
  std::shared_ptr<ResonanceDecayFilterHook> fResonanceDecayFilterHook;

  //PT filter hook
  std::shared_ptr<PTFilterHook> fPTFilterHook;

  //Generic customized hooks vector
  std::shared_ptr<UserHooksVector> fCustomHooksVector;

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

const std::vector<std::string> Pythia8HepMC3Hadronizer::p8SharedResources = {edm::SharedResourceNames::kPythia8};

Pythia8HepMC3Hadronizer::Pythia8HepMC3Hadronizer(const edm::ParameterSet &params)
    : Py8HMC3InterfaceBase(params),
      comEnergy(params.getParameter<double>("comEnergy")),
      LHEInputFileName(params.getUntrackedParameter<std::string>("LHEInputFileName", "")),
      fInitialState(PP),
      UserHooksSet(false),
      nME(-1),
      nMEFiltered(-1),
      nISRveto(0),
      nFSRveto(0) {
  ivhepmc = 3;
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

  // avoid filling weights twice (from v8.30x)
  toHepMC.set_store_weights(false);

  if (params.exists("PhotonFlux")) {
    photonFluxParams = params.getParameter<edm::ParameterSet>("PhotonFlux");
  }

  // Reweight user hook
  //
  if (params.exists("reweightGen")) {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGen";
    edm::ParameterSet rgParams = params.getParameter<edm::ParameterSet>("reweightGen");
    fReweightUserHook = std::make_shared<PtHatReweightUserHook>(rgParams.getParameter<double>("pTRef"),
                                                                rgParams.getParameter<double>("power"));
    edm::LogInfo("Pythia8Interface") << "End setup for reweightGen";
  }
  if (params.exists("reweightGenEmp")) {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGenEmp";
    edm::ParameterSet rgeParams = params.getParameter<edm::ParameterSet>("reweightGenEmp");

    std::string tuneName = "";
    if (rgeParams.exists("tune"))
      tuneName = rgeParams.getParameter<std::string>("tune");
    fReweightEmpUserHook = std::make_shared<PtHatEmpReweightUserHook>(tuneName);
    edm::LogInfo("Pythia8Interface") << "End setup for reweightGenEmp";
  }
  if (params.exists("reweightGenRap")) {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGenRap";
    edm::ParameterSet rgrParams = params.getParameter<edm::ParameterSet>("reweightGenRap");
    fReweightRapUserHook = std::make_shared<RapReweightUserHook>(rgrParams.getParameter<std::string>("yLabSigmaFunc"),
                                                                 rgrParams.getParameter<double>("yLabPower"),
                                                                 rgrParams.getParameter<std::string>("yCMSigmaFunc"),
                                                                 rgrParams.getParameter<double>("yCMPower"),
                                                                 rgrParams.getParameter<double>("pTHatMin"),
                                                                 rgrParams.getParameter<double>("pTHatMax"));
    edm::LogInfo("Pythia8Interface") << "End setup for reweightGenRap";
  }
  if (params.exists("reweightGenPtHatRap")) {
    edm::LogInfo("Pythia8Interface") << "Start setup for reweightGenPtHatRap";
    edm::ParameterSet rgrParams = params.getParameter<edm::ParameterSet>("reweightGenPtHatRap");
    fReweightPtHatRapUserHook =
        std::make_shared<PtHatRapReweightUserHook>(rgrParams.getParameter<std::string>("yLabSigmaFunc"),
                                                   rgrParams.getParameter<double>("yLabPower"),
                                                   rgrParams.getParameter<std::string>("yCMSigmaFunc"),
                                                   rgrParams.getParameter<double>("yCMPower"),
                                                   rgrParams.getParameter<double>("pTHatMin"),
                                                   rgrParams.getParameter<double>("pTHatMax"));
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
      fJetMatchingHook = std::make_shared<JetMatchingHook>(jmParams, &fMasterGen->info);
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
    fEmissionVetoHook1 = std::make_shared<EmissionVetoHook1>(EV1_nFinal,
                                                             EV1_vetoOn,
                                                             EV1_maxVetoCount,
                                                             EV1_pThardMode,
                                                             EV1_pTempMode,
                                                             EV1_emittedMode,
                                                             EV1_pTdefMode,
                                                             EV1_MPIvetoOn,
                                                             EV1_QEDvetoMode,
                                                             EV1_nFinalMode,
                                                             0);
  }

  if (params.exists("UserCustomization")) {
    fCustomHooksVector = std::make_shared<UserHooksVector>();
    const std::vector<edm::ParameterSet> userParams =
        params.getParameter<std::vector<edm::ParameterSet>>("UserCustomization");
    for (const auto &pluginParams : userParams) {
      (fCustomHooksVector->hooks)
          .push_back(
              CustomHookFactory::get()->create(pluginParams.getParameter<std::string>("pluginName"), pluginParams));
    }
  }

  if (params.exists("VinciaPlugin")) {
    throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
        << " Obsolete parameter: VinciaPlugin \n Please use the parameter PartonShowers:model instead \n";
  }
  if (params.exists("DirePlugin")) {
    throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
        << " Obsolete parameter: DirePlugin \n Please use the parameter PartonShowers:model instead \n";
  }
}

bool Pythia8HepMC3Hadronizer::initializeForInternalPartons() {
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

  if (!photonFluxParams.empty()) {
    const auto &beamTypeA = photonFluxParams.getParameter<int>("beamTypeA");
    const auto &beamTypeB = photonFluxParams.getParameter<int>("beamTypeB");
    const auto &radiusA = photonFluxParams.getUntrackedParameter<double>("radiusA", -1.0);
    const auto &radiusB = photonFluxParams.getUntrackedParameter<double>("radiusB", -1.0);
    const auto &zA = photonFluxParams.getUntrackedParameter<int>("zA", -1);
    const auto &zB = photonFluxParams.getUntrackedParameter<int>("zB", -1);
    Pythia8::PDFPtr photonFluxA =
        fMasterGen->settings.flag("PDF:beamA2gamma") ? make_shared<Nucleus2gamma2>(beamTypeA, radiusA, zA) : nullptr;
    Pythia8::PDFPtr photonFluxB =
        fMasterGen->settings.flag("PDF:beamB2gamma") ? make_shared<Nucleus2gamma2>(beamTypeB, radiusB, zB) : nullptr;
    fMasterGen->setPhotonFluxPtr(photonFluxA, photonFluxB);
  }

  if (!fUserHooksVector.get())
    fUserHooksVector = std::make_shared<UserHooksVector>();
  (fUserHooksVector->hooks).clear();

  if (fReweightUserHook.get())
    (fUserHooksVector->hooks).push_back(fReweightUserHook);
  if (fReweightEmpUserHook.get())
    (fUserHooksVector->hooks).push_back(fReweightEmpUserHook);
  if (fReweightRapUserHook.get())
    (fUserHooksVector->hooks).push_back(fReweightRapUserHook);
  if (fReweightPtHatRapUserHook.get())
    (fUserHooksVector->hooks).push_back(fReweightPtHatRapUserHook);
  if (fJetMatchingHook.get())
    (fUserHooksVector->hooks).push_back(fJetMatchingHook);
  if (fEmissionVetoHook1.get()) {
    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook 1 from CMSSW Pythia8Interface";
    (fUserHooksVector->hooks).push_back(fEmissionVetoHook1);
  }

  if (fMasterGen->settings.mode("POWHEG:veto") > 0 || fMasterGen->settings.mode("POWHEG:MPIveto") > 0) {
    if (fJetMatchingHook.get() || fEmissionVetoHook1.get())
      throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
          << " Attempt to turn on PowhegHooks by pythia8 settings but there are incompatible hooks on \n Incompatible "
             "are : jetMatching, emissionVeto1 \n";

    if (!fEmissionVetoHook.get())
      fEmissionVetoHook = std::make_shared<PowhegHooks>();

    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook from pythia8 code";
    (fUserHooksVector->hooks).push_back(fEmissionVetoHook);
  }

  bool PowhegRes = fMasterGen->settings.flag("POWHEGres:calcScales");
  if (PowhegRes) {
    edm::LogInfo("Pythia8Interface") << "Turning on resonance scale setting from CMSSW Pythia8Interface";
    if (!fPowhegResHook.get())
      fPowhegResHook = std::make_shared<PowhegResHook>();
    (fUserHooksVector->hooks).push_back(fPowhegResHook);
  }

  bool PowhegBB4L = fMasterGen->settings.flag("POWHEG:bb4l");
  if (PowhegBB4L) {
    edm::LogInfo("Pythia8Interface") << "Turning on BB4l hook from CMSSW Pythia8Interface";
    if (!fPowhegHooksBB4L.get())
      fPowhegHooksBB4L = std::make_shared<PowhegHooksBB4L>();
    (fUserHooksVector->hooks).push_back(fPowhegHooksBB4L);
  }

  bool TopRecoilHook1 = fMasterGen->settings.flag("TopRecoilHook:doTopRecoilIn");
  if (TopRecoilHook1) {
    edm::LogInfo("Pythia8Interface") << "Turning on RecoilToTop hook from Pythia8Interface";
    if (!fTopRecoilHook.get())
      fTopRecoilHook.reset(new TopRecoilHook());
    (fUserHooksVector->hooks).push_back(fTopRecoilHook);
  }

  //adapted from main89.cc in pythia8 examples
  bool internalMatching = fMasterGen->settings.flag("JetMatching:merge");
  bool internalMerging = !(fMasterGen->settings.word("Merging:Process") == "void");

  if (internalMatching && internalMerging) {
    throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
        << " Only one jet matching/merging scheme can be used at a time. \n";
  }

  if (internalMatching) {
    if (!fJetMatchingPy8InternalHook.get())
      fJetMatchingPy8InternalHook = std::make_shared<Pythia8::JetMatchingMadgraph>();
    (fUserHooksVector->hooks).push_back(fJetMatchingPy8InternalHook);
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
    if (!fMergingHook.get())
      fMergingHook = std::make_shared<Pythia8::amcnlo_unitarised_interface>(scheme);
    (fUserHooksVector->hooks).push_back(fMergingHook);
  }

  bool biasedTauDecayer = fMasterGen->settings.flag("BiasedTauDecayer:filter");
  if (biasedTauDecayer) {
    if (!fBiasedTauDecayer.get()) {
      Pythia8::Info localInfo = fMasterGen->info;
      fBiasedTauDecayer = std::make_shared<BiasedTauDecayer>(&localInfo, &(fMasterGen->settings));
    }
    std::vector<int> handledParticles;
    handledParticles.push_back(15);
    fMasterGen->setDecayPtr(fBiasedTauDecayer, handledParticles);
  }

  bool resonanceDecayFilter = fMasterGen->settings.flag("ResonanceDecayFilter:filter");
  if (resonanceDecayFilter) {
    fResonanceDecayFilterHook = std::make_shared<ResonanceDecayFilterHook>();
    (fUserHooksVector->hooks).push_back(fResonanceDecayFilterHook);
  }

  bool PTFilter = fMasterGen->settings.flag("PTFilter:filter");
  if (PTFilter) {
    fPTFilterHook = std::make_shared<PTFilterHook>();
    (fUserHooksVector->hooks).push_back(fPTFilterHook);
  }

  if (!(fUserHooksVector->hooks).empty() && !UserHooksSet) {
    for (auto &fUserHook : fUserHooksVector->hooks) {
      fMasterGen->addUserHooksPtr(fUserHook);
    }
    UserHooksSet = true;
  }

  if (fCustomHooksVector.get()) {
    edm::LogInfo("Pythia8Interface") << "Adding customized user hooks";
    for (const auto &fUserHook : fCustomHooksVector->hooks) {
      fMasterGen->addUserHooksPtr(fUserHook);
    }
  }

  edm::LogInfo("Pythia8Interface") << "Initializing MasterGen";
  status = fMasterGen->init();

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
    if (!evtgenDecays.get()) {
      evtgenDecays = std::make_shared<EvtGenDecays>(fMasterGen.get(), evtgenDecFile, evtgenPdlFile);
      for (unsigned int i = 0; i < evtgenUserFiles.size(); i++)
        evtgenDecays->readDecayFile(evtgenUserFiles.at(i));
    }
  }

  return (status && status1);
}

bool Pythia8HepMC3Hadronizer::initializeForExternalPartons() {
  edm::LogInfo("Pythia8Interface") << "Initializing for external partons";

  bool status = false, status1 = false;

  if (!fUserHooksVector.get())
    fUserHooksVector = std::make_shared<UserHooksVector>();
  (fUserHooksVector->hooks).clear();

  if (fReweightUserHook.get())
    (fUserHooksVector->hooks).push_back(fReweightUserHook);
  if (fReweightEmpUserHook.get())
    (fUserHooksVector->hooks).push_back(fReweightEmpUserHook);
  if (fReweightRapUserHook.get())
    (fUserHooksVector->hooks).push_back(fReweightRapUserHook);
  if (fReweightPtHatRapUserHook.get())
    (fUserHooksVector->hooks).push_back(fReweightPtHatRapUserHook);
  if (fJetMatchingHook.get())
    (fUserHooksVector->hooks).push_back(fJetMatchingHook);
  if (fEmissionVetoHook1.get()) {
    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook 1 from CMSSW Pythia8Interface";
    (fUserHooksVector->hooks).push_back(fEmissionVetoHook1);
  }

  if (fCustomHooksVector.get()) {
    edm::LogInfo("Pythia8Interface") << "Adding customized user hook";
    for (const auto &fUserHook : fCustomHooksVector->hooks) {
      (fUserHooksVector->hooks).push_back(fUserHook);
    }
  }

  if (fMasterGen->settings.mode("POWHEG:veto") > 0 || fMasterGen->settings.mode("POWHEG:MPIveto") > 0) {
    if (fJetMatchingHook.get() || fEmissionVetoHook1.get())
      throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
          << " Attempt to turn on PowhegHooks by pythia8 settings but there are incompatible hooks on \n Incompatible "
             "are : jetMatching, emissionVeto1 \n";

    if (!fEmissionVetoHook.get())
      fEmissionVetoHook = std::make_shared<PowhegHooks>();

    edm::LogInfo("Pythia8Interface") << "Turning on Emission Veto Hook from pythia8 code";
    (fUserHooksVector->hooks).push_back(fEmissionVetoHook);
  }

  bool PowhegRes = fMasterGen->settings.flag("POWHEGres:calcScales");
  if (PowhegRes) {
    edm::LogInfo("Pythia8Interface") << "Turning on resonance scale setting from CMSSW Pythia8Interface";
    if (!fPowhegResHook.get())
      fPowhegResHook = std::make_shared<PowhegResHook>();
    (fUserHooksVector->hooks).push_back(fPowhegResHook);
  }

  bool PowhegBB4L = fMasterGen->settings.flag("POWHEG:bb4l");
  if (PowhegBB4L) {
    edm::LogInfo("Pythia8Interface") << "Turning on BB4l hook from CMSSW Pythia8Interface";
    if (!fPowhegHooksBB4L.get())
      fPowhegHooksBB4L = std::make_shared<PowhegHooksBB4L>();
    (fUserHooksVector->hooks).push_back(fPowhegHooksBB4L);
  }

  bool TopRecoilHook1 = fMasterGen->settings.flag("TopRecoilHook:doTopRecoilIn");
  if (TopRecoilHook1) {
    edm::LogInfo("Pythia8Interface") << "Turning on RecoilToTop hook from Pythia8Interface";
    if (!fTopRecoilHook.get())
      fTopRecoilHook.reset(new TopRecoilHook());
    (fUserHooksVector->hooks).push_back(fTopRecoilHook);
  }

  //adapted from main89.cc in pythia8 examples
  bool internalMatching = fMasterGen->settings.flag("JetMatching:merge");
  bool internalMerging = !(fMasterGen->settings.word("Merging:Process") == "void");

  if (internalMatching && internalMerging) {
    throw edm::Exception(edm::errors::Configuration, "Pythia8Interface")
        << " Only one jet matching/merging scheme can be used at a time. \n";
  }

  if (internalMatching) {
    if (!fJetMatchingPy8InternalHook.get())
      fJetMatchingPy8InternalHook = std::make_shared<Pythia8::JetMatchingMadgraph>();
    (fUserHooksVector->hooks).push_back(fJetMatchingPy8InternalHook);
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
    if (!fMergingHook.get())
      fMergingHook = std::make_shared<Pythia8::amcnlo_unitarised_interface>(scheme);
    (fUserHooksVector->hooks).push_back(fMergingHook);
  }

  bool biasedTauDecayer = fMasterGen->settings.flag("BiasedTauDecayer:filter");
  if (biasedTauDecayer) {
    if (!fBiasedTauDecayer.get()) {
      Pythia8::Info localInfo = fMasterGen->info;
      fBiasedTauDecayer = std::make_shared<BiasedTauDecayer>(&localInfo, &(fMasterGen->settings));
    }
    std::vector<int> handledParticles;
    handledParticles.push_back(15);
    fMasterGen->setDecayPtr(fBiasedTauDecayer, handledParticles);
  }

  bool resonanceDecayFilter = fMasterGen->settings.flag("ResonanceDecayFilter:filter");
  if (resonanceDecayFilter) {
    fResonanceDecayFilterHook = std::make_shared<ResonanceDecayFilterHook>();
    (fUserHooksVector->hooks).push_back(fResonanceDecayFilterHook);
  }

  bool PTFilter = fMasterGen->settings.flag("PTFilter:filter");
  if (PTFilter) {
    fPTFilterHook = std::make_shared<PTFilterHook>();
    (fUserHooksVector->hooks).push_back(fPTFilterHook);
  }

  if (!(fUserHooksVector->hooks).empty() && !UserHooksSet) {
    for (auto &fUserHook : fUserHooksVector->hooks) {
      fMasterGen->addUserHooksPtr(fUserHook);
    }
    UserHooksSet = true;
  }

  if (!LHEInputFileName.empty()) {
    edm::LogInfo("Pythia8Interface") << "Initialize direct pythia8 reading from LHE file " << LHEInputFileName;
    edm::LogInfo("Pythia8Interface") << "Some LHE information can be not stored";
    fMasterGen->settings.mode("Beams:frameType", 4);
    fMasterGen->settings.word("Beams:LHEF", LHEInputFileName);
    status = fMasterGen->init();

  } else {
    lhaUP = std::make_shared<LHAupLesHouches>();
    lhaUP->setScalesFromLHEF(fMasterGen->settings.flag("Beams:setProductionScalesFromLHEF"));
    lhaUP->loadRunInfo(lheRunInfo());

    if (fJetMatchingHook.get()) {
      fJetMatchingHook->init(lheRunInfo());
    }

    fMasterGen->settings.mode("Beams:frameType", 5);
    fMasterGen->setLHAupPtr(lhaUP);
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
    if (!evtgenDecays.get()) {
      evtgenDecays = std::make_shared<EvtGenDecays>(fMasterGen.get(), evtgenDecFile, evtgenPdlFile);
      for (unsigned int i = 0; i < evtgenUserFiles.size(); i++)
        evtgenDecays->readDecayFile(evtgenUserFiles.at(i));
    }
  }

  return (status && status1);
}

void Pythia8HepMC3Hadronizer::statistics() {
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

bool Pythia8HepMC3Hadronizer::generatePartonsAndHadronize() {
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

  //event() = std::make_unique<HepMC::GenEvent>();
  event3() = std::make_unique<HepMC3::GenEvent>();
  bool py8hepmc = toHepMC.fill_next_event(*(fMasterGen.get()), event3().get());

  if (!py8hepmc) {
    return false;
  }

  // 0th weight is not filled anymore since v8.30x, pushback manually
  event3()->weights().push_back(fMasterGen->info.weight());

  //add ckkw/umeps/unlops merging weight
  if (mergeweight != 1.) {
    event3()->weights()[0] *= mergeweight;
  }

  if (fEmissionVetoHook.get()) {
    nISRveto += fEmissionVetoHook->getNISRveto();
    nFSRveto += fEmissionVetoHook->getNFSRveto();
  }

  //fill additional weights for systematic uncertainties
  if (fMasterGen->info.getWeightsDetailedSize() > 0) {
    for (const string &key : fMasterGen->info.initrwgt->weightsKeys) {
      double wgt = (*fMasterGen->info.weights_detailed)[key];
      event3()->weights().push_back(wgt);
    }
  } else if (fMasterGen->info.getWeightsCompressedSize() > 0) {
    for (unsigned int i = 0; i < fMasterGen->info.getWeightsCompressedSize(); i++) {
      double wgt = fMasterGen->info.getWeightsCompressedValue(i);
      event3()->weights().push_back(wgt);
    }
  }

  // fill shower weights
  // http://home.thep.lu.se/~torbjorn/pythia82html/Variations.html
  if (fMasterGen->info.nWeights() > 1) {
    for (int i = 0; i < fMasterGen->info.nWeights(); ++i) {
      double wgt = fMasterGen->info.weight(i);
      event3()->weights().push_back(wgt);
    }
  }

#if 0
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

    unordered_map<string, double>::iterator it;
    for (it = fDire->weightsPtr->getShowerWeights()->begin(); it != fDire->weightsPtr->getShowerWeights()->end();
         it++) {
      if (it->first == "base")
        continue;
      event()->weights().push_back(it->second);
    }
  }
#endif

  return true;
}

bool Pythia8HepMC3Hadronizer::hadronize() {
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
    event3().reset();
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

  event() = std::make_unique<HepMC::GenEvent>();
  event3() = std::make_unique<HepMC3::GenEvent>();
  bool py8hepmc = toHepMC.fill_next_event(*(fMasterGen.get()), event3().get());

  if (!py8hepmc) {
    return false;
  }

  // 0th weight is not filled anymore since v8.30x, pushback manually
  event3()->weights().push_back(fMasterGen->info.weight());

  //add ckkw/umeps/unlops merging weight
  if (mergeweight != 1.) {
    event3()->weights()[0] *= mergeweight;
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
      event3()->weights().push_back(wgt);
    }
  }

  return true;
}

bool Pythia8HepMC3Hadronizer::residualDecay() {
  Event *pythiaEvent = &(fMasterGen->event);

  int NPartsBeforeDecays = pythiaEvent->size() - 1;  // do NOT count the very 1st "system" particle
                                                     // in Pythia8::Event record; it does NOT even
                                                     // get translated by the HepMCInterface to the
                                                     // HepMC3::GenEvent record !!!

  int NPartsAfterDecays = ((event3().get())->particles()).size();
  if (NPartsAfterDecays == NPartsBeforeDecays)
    return true;

  bool result = true;

  for (const auto &p : (event3().get())->particles()) {
    if (p->id() > NPartsBeforeDecays) {
      if (p->status() == 1 && (fDecayer->particleData).canDecay(p->pid())) {
        fDecayer->event.reset();
        Particle py8part(p->pid(),
                         93,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         p->momentum().x(),
                         p->momentum().y(),
                         p->momentum().z(),
                         p->momentum().t(),
                         p->generated_mass());

        py8part.vProd(p->production_vertex()->position().x(),
                      p->production_vertex()->position().y(),
                      p->production_vertex()->position().z(),
                      p->production_vertex()->position().t());

        py8part.tau((fDecayer->particleData).tau0(p->pid()));
        fDecayer->event.append(py8part);
        int nentries = fDecayer->event.size();
        if (!fDecayer->event[nentries - 1].mayDecay())
          continue;
        result = fDecayer->next();
        int nentries1 = fDecayer->event.size();
        if (nentries1 <= nentries)
          continue;  //same number of particles, no decays...

        p->set_status(2);

        HepMC3::GenVertexPtr prod_vtx0 = make_shared<HepMC3::GenVertex>(  // neglect particle path to decay
            HepMC3::FourVector(p->production_vertex()->position().x(),
                               p->production_vertex()->position().y(),
                               p->production_vertex()->position().z(),
                               p->production_vertex()->position().t()));
        prod_vtx0->add_particle_in(p);
        (event3().get())->add_vertex(prod_vtx0);
        HepMC3::GenParticle *pnew;
        Pythia8::Event pyev = fDecayer->event;
        double momFac = 1.;
        for (int i = 2; i < pyev.size(); ++i) {
          // Fill the particle.
          pnew = new HepMC3::GenParticle(
              HepMC3::FourVector(
                  momFac * pyev[i].px(), momFac * pyev[i].py(), momFac * pyev[i].pz(), momFac * pyev[i].e()),
              pyev[i].id(),
              pyev[i].statusHepMC());
          pnew->set_generated_mass(momFac * pyev[i].m());
          prod_vtx0->add_particle_out(pnew);
        }
      }
    }
  }

  return result;
}

void Pythia8HepMC3Hadronizer::finalizeEvent() {
  bool lhe = lheEvent() != nullptr;

  // protection against empty weight container
  if ((event3()->weights()).empty())
    (event3()->weights()).push_back(1.);

  // now create the GenEventInfo product from the GenEvent and fill
  // the missing pieces
  eventInfo3() = std::make_unique<GenEventInfoProduct3>(event3().get());

  // in pythia pthat is used to subdivide samples into different bins
  // in LHE mode the binning is done by the external ME generator
  // which is likely not pthat, so only filling it for Py6 internal mode
  if (!lhe) {
    eventInfo3()->setBinningValues(std::vector<double>(1, fMasterGen->info.pTHat()));
  }

  eventInfo3()->setDJR(DJR);
  eventInfo3()->setNMEPartons(nME);
  eventInfo3()->setNMEPartonsFiltered(nMEFiltered);

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
      //event3()->print();
      HepMC3::Print::listing(*(event3().get()));
    }
    if (pythiaHepMCVerbosityParticles) {
      std::cout << "Event process = " << fMasterGen->info.code() << "\n"
                << "----------------------" << std::endl;
      //ascii_io->write_event(event().get());
      for (const auto &p : (event3().get())->particles()) {
        HepMC3::Print::line(p, true);
      }
    }
  }
}

std::unique_ptr<GenLumiInfoHeader> Pythia8HepMC3Hadronizer::getGenLumiInfoHeader() const {
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

#if 0
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

    unordered_map<string, double>::iterator it;
    for (it = fDire->weightsPtr->getShowerWeights()->begin(); it != fDire->weightsPtr->getShowerWeights()->end();
         it++) {
      if (it->first == "base")
        continue;
      genLumiInfoHeader->weightNames().push_back(it->first);
    }
  }
#endif

  return genLumiInfoHeader;
}

typedef edm::GeneratorFilter<Pythia8HepMC3Hadronizer, ExternalDecayDriver> Pythia8HepMC3GeneratorFilter;
DEFINE_FWK_MODULE(Pythia8HepMC3GeneratorFilter);

typedef edm::HadronizerFilter<Pythia8HepMC3Hadronizer, ExternalDecayDriver> Pythia8HepMC3HadronizerFilter;
DEFINE_FWK_MODULE(Pythia8HepMC3HadronizerFilter);

typedef edm::ConcurrentGeneratorFilter<Pythia8HepMC3Hadronizer, ConcurrentExternalDecayDriver>
    Pythia8HepMC3ConcurrentGeneratorFilter;
DEFINE_FWK_MODULE(Pythia8HepMC3ConcurrentGeneratorFilter);

typedef edm::ConcurrentHadronizerFilter<Pythia8HepMC3Hadronizer, ConcurrentExternalDecayDriver>
    Pythia8HepMC3ConcurrentHadronizerFilter;
DEFINE_FWK_MODULE(Pythia8HepMC3ConcurrentHadronizerFilter);

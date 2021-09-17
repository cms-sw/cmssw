#include "GeneratorInterface/Pythia8Interface/plugins/JetMatchingHook.h"
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMGFastJet.h"

#include "FWCore/Utilities/interface/Exception.h"

//#include "HepMC/HEPEVT_Wrapper.h"
#include <cassert>

#include "GeneratorInterface/Pythia8Interface/plugins/Py8toJetInput.h"

extern "C" {
// this is patchup for Py6 common block because
// several elements of the VINT array are used in the matching process

extern struct {
  int mint[400];
  double vint[400];
} pyint1_;
}

using namespace gen;
using namespace Pythia8;

JetMatchingHook::JetMatchingHook(const edm::ParameterSet& ps, const Info* info)
    : UserHooks(),
      fRunBlock(nullptr),
      fEventBlock(nullptr),
      fEventNumber(0),
      //      fInfoPtr(info),
      fJetMatching(nullptr),
      fJetInputFill(nullptr),
      fIsInitialized(false) {
  //  assert(fInfoPtr);

  std::string scheme = ps.getParameter<std::string>("scheme");

  if (scheme == "Madgraph") {
    fJetMatching = new JetMatchingMadgraph(ps);
    fJetInputFill = new Py8toJetInputHEPEVT();
  } else if (scheme == "MadgraphFastJet") {
    fJetMatching = new JetMatchingMGFastJet(ps);
    fJetInputFill = new Py8toJetInput();
  } else if (scheme == "MLM" || scheme == "Alpgen") {
    throw cms::Exception("JetMatching") << "Port of " << scheme << "scheme \""
                                        << "\""
                                           " for parton-shower matching is still in progress."
                                        << std::endl;
  } else
    throw cms::Exception("InvalidJetMatching") << "Unknown scheme \"" << scheme
                                               << "\""
                                                  " specified for parton-shower matching."
                                               << std::endl;
}

JetMatchingHook::~JetMatchingHook() {
  if (fJetMatching)
    delete fJetMatching;
}

void JetMatchingHook::init(lhef::LHERunInfo* runInfo) {
  setLHERunInfo(runInfo);
  if (!fRunBlock) {
    throw cms::Exception("JetMatching") << "Invalid RunInfo" << std::endl;
  }
  fJetMatching->init(runInfo);
  double etaMax = fJetMatching->getJetEtaMax();
  fJetInputFill->setJetEtaMax(etaMax);
  return;
}

void JetMatchingHook::beforeHadronization(lhef::LHEEvent* lhee) {
  setLHEEvent(lhee);
  fJetMatching->beforeHadronisation(lhee);

  // here we'll have to adjust, if needed, for "massless" particles
  // from earlier Madgraph version(s)
  // also, we'll have to setup elements of the Py6 fortran array
  // VINT(357), VINT(358), VINT(360) and VINT(390)
  // if ( fJetMatching->getMatchingScheme() == "Madgraph" )
  // {
  //
  // }

  fJetMatching->beforeHadronisationExec();

  return;
}

bool JetMatchingHook::doVetoPartonLevel(const Event& event)
// JetMatchingHook::doVetoPartonLevelEarly( const Event& event )
{
  // event.list();

  // extract "hardest" event - the output will go into workEvent,
  // which is a data mamber of base class UserHooks
  //
  subEvent(event, true);

  if (!hepeup_.nup || fJetMatching->isMatchingDone()) {
    return true;
  }

  //
  // bool jmtch = fJetMatching->match( 0, 0, true ); // true if veto-ed, false if accepted (not veto-ed)
  std::vector<fastjet::PseudoJet> jetInput =
      fJetInputFill->fillJetAlgoInput(event, workEvent, fEventBlock, fJetMatching->getPartonList());
  bool jmtch = fJetMatching->match(fEventBlock, &jetInput);
  if (jmtch) {
    return true;
  }

  // Do not veto events that got this far
  //
  return false;
}

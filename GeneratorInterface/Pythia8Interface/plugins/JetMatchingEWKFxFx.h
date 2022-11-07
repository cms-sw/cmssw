#ifndef GeneratorInterface_Pythia8Interface_JetMatchingEWKFxFx_h
#define GeneratorInterface_Pythia8Interface_JetMatchingEWKFxFx_h

// Class declaration for the new JetMatching plugin
// Author: Carlos Vico (U. Oviedo)
// taken from: https://amcatnlo.web.cern.ch/amcatnlo/JetMatching.h

// Includes
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/JetMatching.h"
#include "Pythia8Plugins/GeneratorInput.h"
#include <memory>
#include "GeneratorInterface/Pythia8Interface/interface/CustomHook.h"

// The plugin must inherit from the original Pythia8::JetMatching
// class

class JetMatchingEWKFxFx : public Pythia8::JetMatching {
public:
  // Constructor and destructor
  JetMatchingEWKFxFx(const edm::ParameterSet& iConfig);
  ~JetMatchingEWKFxFx() override {}

  // Method declaration
  bool initAfterBeams() override;

  bool canVetoPartonLevelEarly() override { return true; }
  bool doVetoPartonLevelEarly(const Pythia8::Event& event) override;

  bool canVetoProcessLevel() override { return true; }
  bool doVetoProcessLevel(Pythia8::Event& event) override;

  // Shower step vetoes (after the first emission, for Shower-kT scheme)
  int numberVetoStep() override { return 1; }
  bool canVetoStep() override { return doShowerKt; }
  bool doVetoStep(int, int, int, const Pythia8::Event&) override;

  Pythia8::SlowJet* slowJetDJR;

  Pythia8::vector<double> getDJR() { return DJR; }
  Pythia8::pair<int, int> nMEpartons() { return nMEpartonsSave; }

  Pythia8::Event getWorkEventJet() { return workEventJetSave; }
  Pythia8::Event getProcessSubset() { return processSubsetSave; }
  bool getExclusive() { return exclusive; }
  double getPTfirst() { return pTfirstSave; }

  // Different steps of the matching algorithm.
  void sortIncomingProcess(const Pythia8::Event&) override;

  void jetAlgorithmInput(const Pythia8::Event&, int) override;
  void runJetAlgorithm() override;
  bool matchPartonsToJets(int) override;
  int matchPartonsToJetsLight() override;
  int matchPartonsToJetsHeavy() override;
  int matchPartonsToJetsOther();
  bool doShowerKtVeto(double pTfirst);

  // Functions to clear and set the jet clustering scales.
  void clearDJR() { DJR.resize(0); }
  void setDJR(const Pythia8::Event& event);
  // Functions to clear and set the jet clustering scales.
  void clear_nMEpartons() { nMEpartonsSave.first = nMEpartonsSave.second = -1; }
  void set_nMEpartons(const int nOrig, const int nMatch) {
    clear_nMEpartons();
    nMEpartonsSave.first = nOrig;
    nMEpartonsSave.second = nMatch;
  };

  // Function to get the current number of partons in the Born state, as
  // read from LHE.
  int npNLO();

private:
  Pythia8::Event processSubsetSave;
  Pythia8::Event workEventJetSave;
  double pTfirstSave;

  bool performVeto;

  Pythia8::vector<int> origTypeIdx[3];
  int nQmatch;
  double qCut, qCutSq, clFact;
  bool doFxFx;
  int nPartonsNow;
  double qCutME, qCutMESq;

  Pythia8::vector<double> DJR;

  Pythia8::pair<int, int> nMEpartonsSave;
};

// Register in the UserHook factory
REGISTER_USERHOOK(JetMatchingEWKFxFx);
#endif

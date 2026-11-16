#ifndef GeneratorInterface_Pythia8Interface_EtaDecayer_h
#define GeneratorInterface_Pythia8Interface_EtaDecayer_h

#include "GeneratorInterface/Pythia8Interface/interface/EtaDecayerModels.h"
#include "Pythia8/Pythia.h"

#include <string>
#include <vector>

// Exclusive eta/eta' decayer: every eta (221) or eta' (331) decayed by Pythia goes to one chosen final state, with
// matrix elements dressed by transition form factors (models and references in EtaDecayerModels.h). The hadronizers
// register it with Pythia8::setDecayPtr when EtaDecayer:filter = on; it cannot be combined with BiasedTauDecayer and
// is not available in the particle guns. The samples give shapes only: apply the branching fraction separately.
//
// Settings:
//   EtaDecayer:filter = on             enable the decayer
//   EtaDecayer:parent = 221 | 331      eta or eta'
//   EtaDecayer:mode   = 2mu, 2egamma, 2mugamma, 2mu2e, 4e, 4mu, 2e2pi, 2mu2pi, 2muA, 2piA, eta2pi, eta2pi0 (eta' only)
//   EtaDecayer:A_id, EtaDecayer:A_type = darkphoton | alp
//                                      2muA, 2piA: the boson A, defined as a Pythia particle
//   EtaDecayer:onePerEvent = off | on  force every parent (default), or one per event chosen at random; with on,
//                                      each event must be weighted by its number of parents
// Daughter order: l-, l+, (gamma); mu-, mu+, e-, e+; l-, l+, l-, l+; l-, l+, pi+, pi-; eta, pi, pi; mu- mu+ or
// pi+ pi-, then A. For the eta' the bounds are maximized over its mass window [mMin, mMax].
class EtaDecayer : public Pythia8::DecayHandler {
public:
  EtaDecayer(Pythia8::Pythia* pythiaPtr, Pythia8::Settings* settingsPtr);

  bool decay(std::vector<int>& idProd,
             std::vector<double>& mProd,
             std::vector<Pythia8::Vec4>& pProd,
             int iDec,
             const Pythia8::Event& event) override;

  // Called after Pythia8::next(): with onePerEvent = on, decays the parents forcing one of them. False on failure.
  bool decayOnePerEvent();

private:
  enum class Mode { TwoMu, LLGamma, MuMuEE, FourL, LLPiPi, EtaPiPi, LLA, PiPiA };

  Pythia8::Pythia* pythia_;
  bool filter_;
  int parentId_;
  bool isEtaPrime_ = false;
  std::string modeName_;
  bool onePerEvent_ = false;
  int forced_ = -1;  // event-record index of the parent to force (onePerEvent)
  Mode mode_ = Mode::TwoMu;
  bool darkPhoton_ = false;  // A_type for 2muA, 2piA
  std::vector<int> daughterIds_;
  std::vector<double> daughterMasses_;
  double threshold_ = 0;
  double muMass_ = 0;                                 // lepton mass of A -> mu+ mu- in the 2piA dark-photon weight
  gen::etadecayer::pipi::Context pipiCtx_;            // pi pi gamma* amplitude inputs (2e2pi, 2mu2pi, 2piA)
  gen::etadecayer::etapipi::Params etaPiPiParams_{};  // Dalitz-plot parameters (eta2pi, eta2pi0)
  double bound_ = 0;                                  // accept-reject bound of the selected decay
};

#endif

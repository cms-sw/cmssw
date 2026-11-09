#ifndef GeneratorInterface_Pythia8Interface_EtaDecayer_h
#define GeneratorInterface_Pythia8Interface_EtaDecayer_h

#include "GeneratorInterface/Pythia8Interface/interface/EtaDecayerModels.h"
#include "Pythia8/Pythia.h"

#include <string>
#include <vector>

// Exclusive eta/eta' decayer: every eta (PDG id 221) or eta' (331) in the event is decayed into one chosen final
// state, with matrix elements dressed by transition form factors (TFF); the models and their references are in
// EtaDecayerModels.h, the samplers in EtaDecayerSamplers.h. It is a Pythia8::DecayHandler that the hadronizers
// Pythia8GeneratorFilter and Pythia8HadronizerFilter (and their HepMC3 versions) register with Pythia8::setDecayPtr
// when EtaDecayer:filter = on. It cannot be combined with BiasedTauDecayer (Pythia8 has a single DecayHandler) and it
// is not available in the Pythia8 particle guns; both configurations are rejected.
//
// The sample is forced-exclusive: it gives the shape of the chosen decay, not its branching fraction, which must be
// applied separately. Radiative (QED) corrections are not included.
//
// Forced parents per event (EtaDecayer:onePerEvent):
//   off (default)  every parent that Pythia decays is forced. An event with N parents is then selected about N times
//                  as often, as in nature for a rare decay (BR * N << 1), so the expected yield is
//                  sigma * L * BR * (fraction of generated events selected). The other parents of a selected event
//                  also decay to the forced final state.
//   on             exactly one parent per event is forced, chosen with equal probability among the parents in the
//                  event record (including those from decays of other hadrons, e.g. eta' -> eta pi pi); the others
//                  decay with Pythia's decay table. This is what list_forced_decays does in the CMS EvtGen interface.
//                  The selection no longer scales with N: each selected event must be weighted by its number of
//                  parents N, and the expected yield is sigma * L * BR * (sum of N over the selected events) /
//                  (number of generated events). Without the weight, events with many parents are under-represented.
//                  The parents are kept stable in Pythia8::next() and decayed afterwards (decayOnePerEvent).
//
// Settings (in the processParameters of the Pythia8 generator):
//   EtaDecayer:filter = on                     enable the decayer
//   EtaDecayer:parent = 221 | 331              eta or eta'
//   EtaDecayer:mode   = 2mu                    P -> mu+ mu-          isotropic
//                       2egamma, 2mugamma      P -> l+ l- gamma      Kroll-Wada with the TFF
//                       2mu2e                  P -> mu+ mu- e+ e-    factorized double-virtual TFF
//                       4e, 4mu                P -> l+ l- l+ l-      direct - exchange interference, TFF applied to
//                                                                    each diagram (for the eta' through its modulus,
//                                                                    as in arXiv:1511.04916, Eq. (23))
//                       2e2pi, 2mu2pi          P -> pi+ pi- l+ l-    magnetic angular term with the mass dependence
//                                                                    of arXiv:2210.14925
//                       eta2pi, eta2pi0        eta' -> eta pi+ pi-,  Dalitz-plot polynomial measured by BESIII
//                                              eta' -> eta pi0 pi0   (arXiv:1709.04627); parent 331 only. The eta
//                                                                    and pi0 are then decayed by Pythia.
//                       2muA                   P -> mu+ mu- A        A = dark photon or ALP
//                       2piA                   P -> pi+ pi- A
//   EtaDecayer:A_id   = <PDG id of A>          2muA, 2piA only
//   EtaDecayer:A_type = darkphoton | alp       2muA, 2piA only
//   EtaDecayer:onePerEvent = off | on          force every parent (default) or one per event, see above
//
// Daughter order in the event record: l-, l+ (then gamma); mu-, mu+, e-, e+ (2mu2e); l-, l+, l-, l+ (4e, 4mu);
// l-, l+, pi+, pi- (2e2pi, 2mu2pi); eta, pi+, pi- (eta2pi); eta, pi0, pi0 (eta2pi0); mu-, mu+, A (2muA);
// pi+, pi-, A (2piA).
//
// The boson A must be defined in the same processParameters as a Pythia particle, e.g. a 0.1 GeV dark photon
// decaying to e+ e-: "9000001:all = A void 3 0 0 0.1 0. 0. 0. 0.", "9000001:addChannel = 1 1. 0 11 -11". A is
// produced on shell at its nominal mass (a width is not sampled); Pythia then decays it with its lifetime and decay
// channels.
//   - Dark photon: couples like a photon (kinetic mixing). The P -> gamma* A vertex has the exact massive-A momentum
//     factor lambda^{3/2}; the TFF is the single-virtual F(q^2) of the l+ l- pair, a leading-order approximation of
//     F(q^2, m_A^2). For 2piA the dark-photon weight is the pi+ pi- l+ l- density at m_ll = m_A for A -> mu+ mu-;
//     its lepton factor is a constant (the production shape does not depend on how A decays), but the mode requires
//     m_A > 2 m_mu.
//   - ALP (pseudoscalar): no production amplitude is implemented; P -> mu+ mu- A and P -> pi+ pi- A follow
//     three-body phase space.
// Mass reach: the decay must be open at the lowest parent mass, m_A < M - 2 m_mu (2muA) or M - 2 m_pi (2piA); with
// the Pythia defaults that is eta 2muA < 0.3365, 2piA < 0.2687, eta' 2muA < 0.7445, 2piA < 0.6766 GeV. For 2muA with a
// dark photon the accept-reject efficiency falls steeply near the endpoint (lambda^{3/2} -> 0): it is usable up to
// about 0.334 GeV (eta) and 0.735 GeV (eta'); beyond that a job can stop with "rejection limit exhausted".
//
// Parent mass: when the parent has a Breit-Wigner mass distribution (the eta' in the Pythia particle data), Pythia
// hands the decayer the sampled mass, so all accept-reject bounds are maximized over the parent mass window. The
// window needs mMax above m0 and mMin (Pythia treats mMax <= mMin as "no upper limit"), and mMin above the decay
// threshold; for a heavy A close to the eta' threshold raise 331:mMin.
//
// The model and the bound range are printed at construction (MessageLogger category "EtaDecayer").
class EtaDecayer : public Pythia8::DecayHandler {
public:
  EtaDecayer(Pythia8::Pythia* pythiaPtr, Pythia8::Settings* settingsPtr);

  bool decay(std::vector<int>& idProd,
             std::vector<double>& mProd,
             std::vector<Pythia8::Vec4>& pProd,
             int iDec,
             const Pythia8::Event& event) override;

  // Called by the hadronizers after Pythia8::next(): with EtaDecayer:onePerEvent = on it decays the parents of the
  // event, forcing one chosen at random; otherwise it does nothing. Returns false if the decays failed.
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

#include "GeneratorInterface/Pythia8Interface/interface/EtaDecayer.h"
#include "GeneratorInterface/Pythia8Interface/interface/EtaDecayerModels.h"
#include "GeneratorInterface/Pythia8Interface/interface/EtaDecayerSamplers.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TLorentzVector.h"

#include <algorithm>
#include <cmath>

namespace {
  template <std::size_t N>
  std::vector<TLorentzVector> toVector(const std::array<TLorentzVector, N>& in) {
    return std::vector<TLorentzVector>(in.begin(), in.end());
  }
}  // namespace

EtaDecayer::EtaDecayer(Pythia8::Pythia* pythiaPtr, Pythia8::Settings* settingsPtr) : pythia_(pythiaPtr) {
  filter_ = settingsPtr->flag("EtaDecayer:filter");
  parentId_ = settingsPtr->mode("EtaDecayer:parent");
  modeName_ = settingsPtr->word("EtaDecayer:mode");
  onePerEvent_ = settingsPtr->flag("EtaDecayer:onePerEvent");
  if (!filter_)
    return;

  if (parentId_ != 221 && parentId_ != 331)
    throw cms::Exception("Configuration") << "EtaDecayer:parent must be 221 (eta) or 331 (eta'), got " << parentId_;
  isEtaPrime_ = parentId_ == 331;

  int aId = 0;
  if (modeName_ == "2muA" || modeName_ == "2piA") {
    aId = settingsPtr->mode("EtaDecayer:A_id");
    const std::string aType = settingsPtr->word("EtaDecayer:A_type");
    if (aId == 0)
      throw cms::Exception("Configuration") << "EtaDecayer:mode = " << modeName_
                                            << " needs EtaDecayer:A_id, the PDG id of A, defined as a Pythia particle";
    if (aType != "darkphoton" && aType != "alp")
      throw cms::Exception("Configuration")
          << "EtaDecayer:A_type must be \"darkphoton\" or \"alp\", got \"" << aType << "\"";
    darkPhoton_ = aType == "darkphoton";
  }

  if (modeName_ == "2mu") {
    mode_ = Mode::TwoMu;
    daughterIds_ = {13, -13};
  } else if (modeName_ == "2mugamma") {
    mode_ = Mode::LLGamma;
    daughterIds_ = {13, -13, 22};
  } else if (modeName_ == "2egamma") {
    mode_ = Mode::LLGamma;
    daughterIds_ = {11, -11, 22};
  } else if (modeName_ == "2mu2e") {
    mode_ = Mode::MuMuEE;
    daughterIds_ = {13, -13, 11, -11};
  } else if (modeName_ == "4e") {
    mode_ = Mode::FourL;
    daughterIds_ = {11, -11, 11, -11};
  } else if (modeName_ == "4mu") {
    mode_ = Mode::FourL;
    daughterIds_ = {13, -13, 13, -13};
  } else if (modeName_ == "2e2pi") {
    mode_ = Mode::LLPiPi;
    daughterIds_ = {11, -11, 211, -211};
  } else if (modeName_ == "2mu2pi") {
    mode_ = Mode::LLPiPi;
    daughterIds_ = {13, -13, 211, -211};
  } else if (modeName_ == "eta2pi" || modeName_ == "eta2pi0") {
    if (!isEtaPrime_)
      throw cms::Exception("Configuration") << "EtaDecayer:mode = " << modeName_ << " needs EtaDecayer:parent = 331";
    mode_ = Mode::EtaPiPi;
    const bool charged = modeName_ == "eta2pi";
    daughterIds_ = charged ? std::vector<int>{221, 211, -211} : std::vector<int>{221, 111, 111};
    etaPiPiParams_ = charged ? gen::etadecayer::etapipi::kCharged : gen::etadecayer::etapipi::kNeutral;
  } else if (modeName_ == "2muA") {
    mode_ = Mode::LLA;
    daughterIds_ = {13, -13, aId};
  } else if (modeName_ == "2piA") {
    mode_ = Mode::PiPiA;
    daughterIds_ = {211, -211, aId};
  } else
    throw cms::Exception("Configuration") << "Unknown EtaDecayer:mode \"" << modeName_ << "\"";

  auto& pd = pythia_->particleData;
  for (int id : daughterIds_) {
    const double m = pd.m0(id);
    daughterMasses_.push_back(m);
    threshold_ += m;
  }
  const double mA = aId != 0 ? daughterMasses_[2] : 0.;
  if (aId != 0 && !(mA > 0))
    throw cms::Exception("Configuration") << "EtaDecayer:A_id = " << aId
                                          << " has no mass: define A as a Pythia particle (\"<id>:all = ...\") "
                                             "in the same processParameters";
  muMass_ = pd.m0(13);
  if (mode_ == Mode::PiPiA && darkPhoton_ && !(mA > 2 * muMass_))
    throw cms::Exception("Configuration") << "EtaDecayer:mode = 2piA with a dark photon assumes A -> mu+ mu- and "
                                             "needs m_A > 2 m_mu, got m_A = "
                                          << mA << " GeV";
  const double m0 = pd.m0(parentId_);
  if (!(m0 > threshold_))
    throw cms::Exception("Configuration") << "EtaDecayer: parent mass " << m0 << " GeV is below the threshold "
                                          << threshold_ << " GeV of EtaDecayer:mode = " << modeName_;

  // Bounds are maximized over m0 and, for a parent with a width, 9 masses across [mMin, mMax].
  std::vector<double> boundMasses = {m0};
  if (pd.mWidth(parentId_) > 0) {
    const double lo = pd.mMin(parentId_), hi = pd.mMax(parentId_);
    if (!(hi > lo))
      throw cms::Exception("Configuration")
          << "EtaDecayer: parent " << parentId_ << " has a width but no finite upper mass limit (mMin = " << lo
          << ", mMax = " << hi << " GeV); set " << parentId_ << ":mMax above m0 = " << m0 << " GeV";
    if (!(hi > m0))
      throw cms::Exception("Configuration") << "EtaDecayer: parent " << parentId_ << " mass window [" << lo << ", "
                                            << hi << "] GeV must have mMax above m0 = " << m0 << " GeV";
    if (!(lo > threshold_))
      throw cms::Exception("Configuration")
          << "EtaDecayer: parent " << parentId_ << " mass window starts at mMin = " << lo
          << " GeV, at or below the threshold " << threshold_ << " GeV of EtaDecayer:mode = " << modeName_ << "; raise "
          << parentId_ << ":mMin";
    for (int k = 0; k <= 8; ++k)
      boundMasses.push_back(lo + (hi - lo) * k / 8.);
  }
  const auto overMasses = [&boundMasses](auto f) {
    double b = 0;
    for (double m : boundMasses)
      b = std::max(b, static_cast<double>(f(m)));
    return b;
  };

  using namespace gen::etadecayer;
  // rho(770) and rho(1450) of the pi pi gamma* amplitude, from the Pythia particle data
  auto& ctx = pipiCtx_;
  ctx.isEtaPrime = isEtaPrime_;
  ctx.rhoMass = pd.m0(113);
  ctx.rhoGamma = pd.mWidth(113);
  ctx.rhopMass = pd.m0(100113);
  ctx.rhopGamma = pd.mWidth(100113);
  const double ml = daughterMasses_[0];
  switch (mode_) {
    case Mode::TwoMu:
      bound_ = 1.;
      break;
    case Mode::LLGamma:
      bound_ =
          overMasses([&](double mP) { return isEtaPrime_ ? etaPrimeSingleBound(mP, ml) : etaSingleBound(mP, ml); });
      break;
    case Mode::MuMuEE:
    case Mode::FourL: {
      const double ml2 = daughterMasses_[2];
      bound_ = overMasses(
          [&](double mP) { return isEtaPrime_ ? etaPrimeDoubleBound(mP, ml, ml2) : etaDoubleBound(mP, ml, ml2); });
      break;
    }
    case Mode::LLPiPi:
      ctx.mPi = daughterMasses_[2];
      bound_ = overMasses([&](double mP) { return llpipiBound(mP, ml, ctx); });
      break;
    case Mode::EtaPiPi:
      bound_ = overMasses(
          [&](double mP) { return etaPiPiBound(mP, daughterMasses_[0], daughterMasses_[1], etaPiPiParams_); });
      break;
    case Mode::LLA:
      bound_ = overMasses([&](double mP) {
        return darkPhoton_ ? darkPhotonLLABound(mP, ml, mA, isEtaPrime_) : alpLLABound(mP, ml, mA);
      });
      break;
    case Mode::PiPiA:
      ctx.mPi = daughterMasses_[0];
      bound_ = overMasses([&](double mP) {
        return darkPhoton_ ? darkPhotonPiPiABound(mP, mA, muMass_, ctx) : alpPiPiABound(mP, ctx.mPi, mA);
      });
      break;
  }
  if (!(bound_ > 0) || !std::isfinite(bound_))
    throw cms::Exception("EtaDecayer") << "invalid accept-reject bound " << bound_
                                       << " for EtaDecayer:mode = " << modeName_;

  edm::LogVerbatim("EtaDecayer") << "parent " << parentId_ << " forced to EtaDecayer:mode = " << modeName_
                                 << (aId != 0 ? (darkPhoton_ ? " (dark photon" : " (ALP") + std::string(", PDG id ") +
                                                    std::to_string(aId) + ", m = " + std::to_string(mA) + " GeV)"
                                              : std::string())
                                 << "; bounds maximized over parent masses ["
                                 << *std::min_element(boundMasses.begin(), boundMasses.end()) << ", "
                                 << *std::max_element(boundMasses.begin(), boundMasses.end()) << "] GeV"
                                 << (onePerEvent_ ? "; one parent per event is forced" : "; every parent is forced");

  // onePerEvent: the parents stay stable in Pythia8::next() and are decayed in decayOnePerEvent()
  if (onePerEvent_)
    pd.mayDecay(parentId_, false);
}

bool EtaDecayer::decayOnePerEvent() {
  if (!filter_ || !onePerEvent_)
    return true;
  auto& event = pythia_->event;
  std::vector<int> candidates;
  for (int i = 0; i < event.size(); ++i)
    if (event[i].id() == parentId_ && event[i].isFinal())
      candidates.push_back(i);
  if (candidates.empty())
    return true;
  const auto n = static_cast<int>(candidates.size());
  forced_ = candidates[std::min(n - 1, static_cast<int>(n * pythia_->rndm.flat()))];
  // decay() forces the chosen parent and returns false for the others, which Pythia then decays itself
  pythia_->particleData.mayDecay(parentId_, true);
  const bool ok = pythia_->moreDecays();
  pythia_->particleData.mayDecay(parentId_, false);
  forced_ = -1;
  return ok;
}

bool EtaDecayer::decay(std::vector<int>& idProd,
                       std::vector<double>& mProd,
                       std::vector<Pythia8::Vec4>& pProd,
                       int iDec,
                       const Pythia8::Event& /*event*/) {
  if (!filter_ || (onePerEvent_ && iDec != forced_))
    return false;
  if (idProd.size() != 1 || mProd.size() != 1 || pProd.size() != 1 || idProd.front() != parentId_)
    return false;

  const auto& p = pProd.front();
  const TLorentzVector parent(p.px(), p.py(), p.pz(), p.e());
  if (parent.M() <= threshold_)
    throw cms::Exception("EtaDecayer") << "parent mass " << parent.M() << " GeV is below the decay threshold "
                                       << threshold_ << " GeV";

  using namespace gen::etadecayer;
  auto flat = [this]() { return pythia_->rndm.flat(); };
  const double bound = bound_, ml = daughterMasses_[0];
  const bool etaPrime = isEtaPrime_;
  std::vector<TLorentzVector> result;
  switch (mode_) {
    case Mode::TwoMu:
      result = toVector(twoBody(parent, ml, flat));
      break;
    case Mode::LLGamma:
      result = toVector(singleDalitz(
          parent,
          ml,
          [etaPrime, bound](double q2) {
            return etaPrime ? etaPrimeFormFactorNormSq(q2) / bound : std::pow(etaFormFactorNorm(q2), 2) / bound;
          },
          flat));
      break;
    case Mode::MuMuEE:
      result = toVector(doubleDalitz(
          parent,
          ml,
          daughterMasses_[2],
          [etaPrime, bound](double s1, double s2) {
            return etaPrime ? etaPrimeFormFactorNormSq(s1) * etaPrimeFormFactorNormSq(s2) / bound
                            : std::pow(etaFormFactorNorm(s1), 2) * std::pow(etaFormFactorNorm(s2), 2) / bound;
          },
          flat));
      break;
    case Mode::FourL:
      // F(s1, s2) = F(s1) F(s2); for the eta' the TFF enters through its modulus (EtaDecayerModels.h).
      result = toVector(identicalDoubleDalitz(
          parent,
          ml,
          [etaPrime](double s1, double s2) {
            return etaPrime ? Complex(std::sqrt(etaPrimeFormFactorNormSq(s1)) * std::sqrt(etaPrimeFormFactorNormSq(s2)))
                            : Complex(etaFormFactorNorm(s1) * etaFormFactorNorm(s2));
          },
          bound,
          flat));
      break;
    case Mode::LLPiPi: {
      const auto& ctx = pipiCtx_;
      result = toVector(llpipi(
          parent,
          ml,
          ctx.mPi,
          [&ctx, bound](double s, double sl) { return pipi::massWeight(s, sl, ctx) / bound; },
          flat));
      break;
    }
    case Mode::EtaPiPi:
      result = toVector(etaPiPi(parent, daughterMasses_[0], daughterMasses_[1], etaPiPiParams_, bound, flat));
      break;
    case Mode::LLA: {
      const double mA = daughterMasses_[2], M2 = parent.M() * parent.M(), r = mA * mA / M2;
      if (darkPhoton_)
        // beta lambda^{3/2} (angular factor) |F|^2; the 1/x pole cancels against the log-uniform proposal.
        result = toVector(llA(
            parent,
            ml,
            mA,
            [&](double x, double c) {
              const double beta2 = 1 - 4 * ml * ml / (x * M2);
              const double beta = std::sqrt(std::max(beta2, 0.));
              const double ang = (1 + c * c + (1 - beta2) * (1 - c * c)) / 2;
              return beta * darkPhotonVertexFactor(x, r) * ang * formFactorNormSq(x * M2, etaPrime) / bound;
            },
            flat));
      else
        // three-body phase space: beta p, times x from the log-uniform proposal
        result = toVector(llA(
            parent,
            ml,
            mA,
            [&](double x, double /*c*/) {
              const double beta2 = 1 - 4 * ml * ml / (x * M2);
              if (!(beta2 > 0))
                return 0.;
              const double pA = splitMomentum(parent.M(), parent.M() * std::sqrt(x), mA);
              if (!(pA > 0))
                return 0.;
              return std::sqrt(beta2) * pA * x / bound;
            },
            flat));
      break;
    }
    case Mode::PiPiA: {
      const double mA = daughterMasses_[2];
      if (darkPhoton_) {
        result = toVector(pipiADarkPhoton(parent, mA, muMass_, pipiCtx_, bound, flat));
      } else
        result = toVector(pipiAPhaseSpace(parent, daughterMasses_[0], mA, bound, flat));
      break;
    }
  }

  TLorentzVector sum;
  for (const auto& daughter : result)
    sum += daughter;
  const auto residual = sum - parent;
  if (std::abs(residual.E()) + residual.Vect().Mag() > 1e-7 * std::max(1., parent.E()))
    throw cms::Exception("EtaDecayer") << "decay products do not conserve four-momentum";

  for (unsigned i = 0; i < daughterIds_.size(); ++i) {
    idProd.push_back(daughterIds_[i]);
    mProd.push_back(result[i].M());
    pProd.emplace_back(result[i].Px(), result[i].Py(), result[i].Pz(), result[i].E());
  }
  return true;
}

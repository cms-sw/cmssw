#include "catch2/catch_all.hpp"

#include "FWCore/Utilities/interface/Exception.h"
#include "GeneratorInterface/Pythia8Interface/interface/EtaDecayer.h"
#include "GeneratorInterface/Pythia8Interface/interface/EtaDecayerModels.h"
#include "GeneratorInterface/Pythia8Interface/interface/EtaDecayerSamplers.h"

#include "Pythia8/Pythia.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

static constexpr auto s_tag = "[EtaDecayer]";

namespace {
  // Pythia with the EtaDecayer settings registered as in Py8InterfaceBase.cc, and the given commands applied.
  std::unique_ptr<Pythia8::Pythia> makePythia(const std::vector<std::string>& commands) {
    // the xmldoc path is overridden by PYTHIA8DATA, set in the CMSSW environment
    auto pythia = std::make_unique<Pythia8::Pythia>("../share/Pythia8/xmldoc", false);
    auto& s = pythia->settings;
    s.addFlag("EtaDecayer:filter", false);
    s.addMode("EtaDecayer:parent", 221, false, false, 0, 0);
    s.addWord("EtaDecayer:mode", "2mu2e");
    s.addMode("EtaDecayer:A_id", 0, false, false, 0, 0);
    s.addWord("EtaDecayer:A_type", "darkphoton");
    s.addFlag("EtaDecayer:onePerEvent", false);
    for (const auto& c : commands)
      pythia->readString(c);
    pythia->rndm.init(12345);
    return pythia;
  }

  std::vector<std::string> commands(int parent, const std::string& mode) {
    return {"EtaDecayer:filter = on", "EtaDecayer:parent = " + std::to_string(parent), "EtaDecayer:mode = " + mode};
  }

  // Boson A with mass mA (spin type 3 = vector, 1 = scalar), PDG id 9000001.
  std::vector<std::string> withBoson(std::vector<std::string> c, const std::string& type, double mA) {
    c.push_back("9000001:all = A void " + std::string(type == "darkphoton" ? "3" : "1") + " 0 0 " + std::to_string(mA) +
                " 0. 0. 0. 0.");
    c.push_back("EtaDecayer:A_id = 9000001");
    c.push_back("EtaDecayer:A_type = " + type);
    return c;
  }

  double invMass(const Pythia8::Vec4& a, const Pythia8::Vec4& b) { return (a + b).mCalc(); }

  // Decays nDecay parents (alternately at rest and boosted) and checks the daughters.
  void checkDecays(Pythia8::Pythia& pythia, int parent, const std::vector<int>& expectedIds, int nDecay = 500) {
    EtaDecayer decayer(&pythia, &pythia.settings);
    // Parent masses: m0, or for a parent with a mass spread (the eta') values across [mMin, mMax].
    const auto& pd = pythia.particleData;
    const bool spread = pd.mWidth(parent) > 0;
    Pythia8::Event dummy;
    for (int i = 0; i < nDecay; ++i) {
      const double M = spread ? pd.mMin(parent) + (pd.mMax(parent) - pd.mMin(parent)) * (i % 11) / 10. : pd.m0(parent);
      Pythia8::Vec4 p(0, 0, 0, M);
      if (i % 2)
        p = Pythia8::Vec4(1.2, -0.7, 8.0, std::sqrt(M * M + 1.44 + 0.49 + 64.));
      std::vector<int> idProd = {parent};
      std::vector<double> mProd = {M};
      std::vector<Pythia8::Vec4> pProd = {p};
      REQUIRE(decayer.decay(idProd, mProd, pProd, 0, dummy));
      REQUIRE(idProd.size() == expectedIds.size() + 1);
      Pythia8::Vec4 sum;
      for (unsigned k = 0; k < expectedIds.size(); ++k) {
        REQUIRE(idProd[k + 1] == expectedIds[k]);
        const double m0 = pythia.particleData.m0(expectedIds[k]);
        REQUIRE(std::abs(pProd[k + 1].mCalc() - m0) < 1e-6 * std::max(1., pProd[k + 1].e()));
        sum += pProd[k + 1];
      }
      REQUIRE(std::abs(sum.e() - p.e()) < 1e-7 * p.e());
      REQUIRE(std::abs(sum.px() - p.px()) < 1e-7 * p.e());
      REQUIRE(std::abs(sum.py() - p.py()) < 1e-7 * p.e());
      REQUIRE(std::abs(sum.pz() - p.pz()) < 1e-7 * p.e());
      // Opposite-sign pairs (first two daughters) are above threshold and below the kinematic limit.
      const double m12 = invMass(pProd[1], pProd[2]);
      REQUIRE(m12 >= pythia.particleData.m0(expectedIds[0]) + pythia.particleData.m0(expectedIds[1]) - 1e-9);
      REQUIRE(m12 <= M + 1e-9);
    }
    // A different particle is not handled.
    std::vector<int> idOther = {111};
    std::vector<double> mOther = {0.135};
    std::vector<Pythia8::Vec4> pOther = {Pythia8::Vec4(0, 0, 0, 0.135)};
    REQUIRE_FALSE(decayer.decay(idOther, mOther, pOther, 0, dummy));
  }
}  // namespace

TEST_CASE("EtaDecayer form factors and vertex factors", s_tag) {
  using namespace gen::etadecayer;
  SECTION("normalization") {
    REQUIRE(etaFormFactorNorm(0.) == Catch::Approx(1.).epsilon(1e-12));
    REQUIRE(etaPrimeFormFactorNormSq(0.) == Catch::Approx(1.).epsilon(1e-12));
  }
  SECTION("eta slope b_eta = M_eta^2 F'(0)/F(0) = 0.5749 (arXiv:1504.07742, Eq. (A.2))") {
    const double h = 1e-5, mEta = 0.547862;
    const double slope = (etaFormFactorNorm(h) - etaFormFactorNorm(-h)) / (2 * h);
    REQUIRE(mEta * mEta * slope == Catch::Approx(0.5749).epsilon(1e-3));
  }
  SECTION("eta' TFF is continuous at the 0.70 GeV matching point") {
    const double q2 = etaprime_tff::matchQ2, e = 1e-9;
    REQUIRE(etaPrimeFormFactorNormSq(q2 + e) == Catch::Approx(etaPrimeFormFactorNormSq(q2 - e)).epsilon(1e-6));
  }
  SECTION("dark-photon vertex factor reduces to (1-x)^3 for m_A -> 0") {
    for (double x : {0.01, 0.2, 0.5, 0.9})
      REQUIRE(darkPhotonVertexFactor(x, 0.) == Catch::Approx(std::pow(1 - x, 3)).epsilon(1e-12));
    REQUIRE(darkPhotonVertexFactor(0.5, 0.2) == 0.);  // sqrt(x) + sqrt(r) > 1: closed
  }
}

TEST_CASE("EtaDecayer decays", s_tag) {
  for (int parent : {221, 331}) {
    SECTION("SM modes, parent " + std::to_string(parent)) {
      const std::vector<std::pair<std::string, std::vector<int>>> modes = {{"2mu", {13, -13}},
                                                                           {"2egamma", {11, -11, 22}},
                                                                           {"2mugamma", {13, -13, 22}},
                                                                           {"2mu2e", {13, -13, 11, -11}},
                                                                           {"4e", {11, -11, 11, -11}},
                                                                           {"4mu", {13, -13, 13, -13}},
                                                                           {"2e2pi", {11, -11, 211, -211}},
                                                                           {"2mu2pi", {13, -13, 211, -211}}};
      for (const auto& [mode, ids] : modes) {
        auto pythia = makePythia(commands(parent, mode));
        checkDecays(*pythia, parent, ids);
      }
    }
    if (parent == 331)
      SECTION("eta' -> eta pi pi") {
        auto pCharged = makePythia(commands(parent, "eta2pi"));
        checkDecays(*pCharged, parent, {221, 211, -211});
        auto pNeutral = makePythia(commands(parent, "eta2pi0"));
        checkDecays(*pNeutral, parent, {221, 111, 111});
      }
    SECTION("BSM modes, parent " + std::to_string(parent)) {
      for (const std::string type : {"darkphoton", "alp"}) {
        auto pLLA = makePythia(withBoson(commands(parent, "2muA"), type, 0.1));
        checkDecays(*pLLA, parent, {13, -13, 9000001});
        auto pPiPiA = makePythia(withBoson(commands(parent, "2piA"), type, 0.25));
        checkDecays(*pPiPiA, parent, {211, -211, 9000001});
      }
      if (parent == 331) {  // heavy A: the eta' mass window starts just above the threshold
        auto pHeavy = makePythia(withBoson(commands(parent, "2piA"), "darkphoton", 0.67));
        checkDecays(*pHeavy, parent, {211, -211, 9000001});
      }
    }
  }
}

// The sampled Dalitz plot reproduces the model: <Y> and <X^2> of the accepted decays against their expectation from a
// numerical integration of |M(X, Y)|^2 over the phase space (flat in s1, s2).
TEST_CASE("EtaDecayer eta' -> eta pi pi Dalitz plot", s_tag) {
  using namespace gen::etadecayer;
  const double M = 0.95778, mEta = 0.547862;
  for (const auto& [mPi, params] :
       {std::pair{0.13957039, etapipi::kCharged}, std::pair{0.1349768, etapipi::kNeutral}}) {
    const double Q = M - mEta - 2 * mPi;
    const auto xy = [&](const etapipi::Kinematics& k) {
      return std::pair{std::sqrt(3.) * (k.e1 - k.e2) / Q, (mEta + 2 * mPi) / mPi * (k.eEta - mEta) / Q - 1};
    };
    // expectation
    const double lo = std::pow(mEta + mPi, 2), hi = std::pow(M - mPi, 2);
    const unsigned grid = 1000;
    double sw = 0, swY = 0, swX2 = 0, n0 = 0, sY0 = 0;
    etapipi::Kinematics k;
    for (unsigned i = 0; i < grid; ++i)
      for (unsigned j = 0; j < grid; ++j)
        if (etapipi::kinematics(
                M, mEta, mPi, lo + (hi - lo) * (i + 0.5) / grid, lo + (hi - lo) * (j + 0.5) / grid, k)) {
          const double w = etapipi::weight(M, mEta, mPi, k, params);
          const auto [X, Y] = xy(k);
          sw += w;
          swY += w * Y;
          swX2 += w * X * X;
          n0 += 1;
          sY0 += Y;
        }
    // sample
    const double bound = etaPiPiBound(M, mEta, mPi, params);
    Pythia8::Rndm rndm(4711);
    auto flat = [&rndm]() { return rndm.flat(); };
    const TLorentzVector parent(0.3, -0.2, 2.0, std::sqrt(M * M + 0.09 + 0.04 + 4.0));
    const int n = 200000;
    double sY = 0, sY2 = 0, sX2 = 0, sX4 = 0;
    for (int i = 0; i < n; ++i) {
      auto out = etaPiPi(parent, mEta, mPi, params, bound, flat);
      for (auto& p : out)
        p.Boost(-parent.BoostVector());
      const double X = std::sqrt(3.) * (out[1].E() - out[2].E()) / Q;
      const double Y = (mEta + 2 * mPi) / mPi * (out[0].E() - mEta) / Q - 1;
      sY += Y;
      sY2 += Y * Y;
      sX2 += X * X;
      sX4 += X * X * X * X;
    }
    const double meanY = sY / n, meanX2 = sX2 / n;
    const double errY = std::sqrt((sY2 / n - meanY * meanY) / n), errX2 = std::sqrt((sX4 / n - meanX2 * meanX2) / n);
    REQUIRE(std::abs(meanY - swY / sw) < 4 * errY);
    REQUIRE(std::abs(meanX2 - swX2 / sw) < 4 * errX2);
    // the model differs from phase space by much more than the test resolution
    REQUIRE(std::abs(swY / sw - sY0 / n0) > 10 * errY);
  }
}

// Minimum-bias events as in the hadronizers (setDecayPtr, then decayOnePerEvent after next()). Returns, over the
// events with at least one eta, the numbers of forced (mu- mu+ e- e+) and of other eta decays.
namespace {
  std::pair<int, int> countForced(bool onePerEvent, int nEvent = 20) {
    auto c = commands(221, "2mu2e");
    c.insert(c.end(), {"SoftQCD:nonDiffractive = on", "Beams:eCM = 13600.", "Print:quiet = on"});
    c.push_back(std::string("EtaDecayer:onePerEvent = ") + (onePerEvent ? "on" : "off"));
    auto pythia = makePythia(c);
    auto decayer = std::make_shared<EtaDecayer>(pythia.get(), &pythia->settings);
    pythia->setDecayPtr(decayer, {221});
    REQUIRE(pythia->init());
    int nForced = 0, nOther = 0;
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
      REQUIRE(pythia->next());
      REQUIRE(decayer->decayOnePerEvent());
      int nForcedEvent = 0, nEta = 0;
      for (int i = 0; i < pythia->event.size(); ++i) {
        const auto& eta = pythia->event[i];
        if (eta.id() != 221)
          continue;
        ++nEta;
        REQUIRE_FALSE(eta.isFinal());  // every eta is decayed
        std::vector<int> ids;
        for (int d : eta.daughterList())
          ids.push_back(pythia->event[d].id());
        if (ids == std::vector<int>{13, -13, 11, -11})
          ++nForcedEvent;
      }
      if (nEta == 0)
        continue;
      if (onePerEvent)
        REQUIRE(nForcedEvent == 1);
      nForced += nForcedEvent;
      nOther += nEta - nForcedEvent;
    }
    return {nForced, nOther};
  }
}  // namespace

TEST_CASE("EtaDecayer forced parents per event", s_tag) {
  SECTION("every eta forced (default)") {
    const auto [nForced, nOther] = countForced(false);
    REQUIRE(nForced > 0);
    REQUIRE(nOther == 0);
  }
  SECTION("one eta per event forced, the others decayed by Pythia") {
    const auto [nForced, nOther] = countForced(true);
    REQUIRE(nForced > 0);
    REQUIRE(nOther > 0);
  }
}

TEST_CASE("EtaDecayer configuration errors", s_tag) {
  SECTION("disabled decayer does nothing") {
    auto pythia = makePythia({});
    EtaDecayer decayer(pythia.get(), &pythia->settings);
    std::vector<int> id = {221};
    std::vector<double> m = {0.547862};
    std::vector<Pythia8::Vec4> p = {Pythia8::Vec4(0, 0, 0, 0.547862)};
    Pythia8::Event dummy;
    REQUIRE_FALSE(decayer.decay(id, m, p, 0, dummy));
  }
  SECTION("unknown mode") {
    auto pythia = makePythia(commands(221, "3mu"));
    REQUIRE_THROWS_AS(EtaDecayer(pythia.get(), &pythia->settings), cms::Exception);
  }
  SECTION("2muA without A") {
    auto pythia = makePythia(commands(221, "2muA"));
    REQUIRE_THROWS_AS(EtaDecayer(pythia.get(), &pythia->settings), cms::Exception);
  }
  SECTION("eta' -> eta pi pi with the eta as parent") {
    auto pythia = makePythia(commands(221, "eta2pi"));
    REQUIRE_THROWS_AS(EtaDecayer(pythia.get(), &pythia->settings), cms::Exception);
  }
  SECTION("A too heavy") {
    auto pythia = makePythia(withBoson(commands(221, "2muA"), "darkphoton", 0.4));
    REQUIRE_THROWS_AS(EtaDecayer(pythia.get(), &pythia->settings), cms::Exception);
  }
  SECTION("parent outside 221, 331") {
    auto pythia = makePythia(commands(111, "2mu"));
    REQUIRE_THROWS_AS(EtaDecayer(pythia.get(), &pythia->settings), cms::Exception);
  }
  SECTION("A without a mass") {
    auto c = commands(221, "2muA");
    c.push_back("EtaDecayer:A_id = 9000002");
    auto pythia = makePythia(c);
    REQUIRE_THROWS_AS(EtaDecayer(pythia.get(), &pythia->settings), cms::Exception);
  }
  SECTION("eta' mass window without a finite upper limit") {
    auto c = commands(331, "4e");
    c.push_back("331:mMax = 0.");
    auto pythia = makePythia(c);
    REQUIRE_THROWS_AS(EtaDecayer(pythia.get(), &pythia->settings), cms::Exception);
  }
  SECTION("2piA dark photon below the A -> mu+ mu- threshold") {
    auto pythia = makePythia(withBoson(commands(221, "2piA"), "darkphoton", 0.15));
    REQUIRE_THROWS_AS(EtaDecayer(pythia.get(), &pythia->settings), cms::Exception);
  }
  SECTION("eta' mass window reaching below the threshold") {
    auto c = withBoson(commands(331, "2piA"), "alp", 0.67);
    c.push_back("331:mMin = 0.90");
    auto pythia = makePythia(c);
    REQUIRE_THROWS_AS(EtaDecayer(pythia.get(), &pythia->settings), cms::Exception);
  }
}

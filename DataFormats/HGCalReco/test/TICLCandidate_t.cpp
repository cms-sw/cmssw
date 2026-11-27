#include <array>
#include <cmath>
#include <limits>

#include <catch2/catch_all.hpp>

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

namespace {
  ticl::Trackster tracksterWithTime(float time, float timeError, float energy) {
    ticl::Trackster ts;
    ts.setTimeAndError(time, timeError);
    ts.setRegressedEnergy(energy);
    ts.setRawEnergy(energy);
    ts.setBarycenter({10.f, 0.f, 330.f});
    return ts;
  }
}  // namespace

TEST_CASE("appendTrackster keeps the candidate time, addTrackster takes the trackster time", "[TICLCandidate]") {
  const auto seed = tracksterWithTime(0.10f, 0.02f, 20.f);
  const auto claim = tracksterWithTime(-1.f, -1.f, 2.f);
  edm::Ptr<reco::Track> noTrack;
  TICLCandidate cand(noTrack, edm::Ptr<ticl::Trackster>(&seed, 0));
  // The MTD time of the candidate.
  cand.setTime(0.35f, 0.03f);
  cand.setMTDTime(0.35f, 0.03f);

  SECTION("appendTrackster") {
    cand.appendTrackster(edm::Ptr<ticl::Trackster>(&claim, 1));
    REQUIRE(cand.tracksters().size() == 2);
    REQUIRE(cand.tracksters()[1].key() == 1);
    REQUIRE(cand.time() == 0.35f);
    REQUIRE(cand.timeError() == 0.03f);
    REQUIRE(cand.MTDtime() == 0.35f);
  }
  SECTION("addTrackster") {
    cand.addTrackster(edm::Ptr<ticl::Trackster>(&claim, 1));
    REQUIRE(cand.tracksters().size() == 2);
    REQUIRE(cand.time() == -1.f);
    REQUIRE(cand.timeError() == -1.f);
  }
}

TEST_CASE("TICLCandidate stores a probability as a 16-bit integer with 65535 for 1", "[TICLCandidate]") {
  using C = io_v1::TICLCandidate;
  REQUIRE(C::encodeProbability(0.f) == 0);
  REQUIRE(C::encodeProbability(1.f) == 65535);
  REQUIRE(C::encodeProbability(0.5f) == 32768);
  REQUIRE(C::encodeProbability(-0.2f) == 0);
  REQUIRE(C::encodeProbability(1.5f) == 65535);
  REQUIRE(C::encodeProbability(std::numeric_limits<float>::quiet_NaN()) == 0);
  REQUIRE(C::decodeProbability(0) == 0.f);
  REQUIRE(C::decodeProbability(65535) == 1.f);

  const float maxError = 0.5f / 65535.f + 1e-7f;
  for (int i = 0; i <= 100000; ++i) {
    const float p = i / 100000.f;
    REQUIRE(std::abs(C::decodeProbability(C::encodeProbability(p)) - p) <= maxError);
  }
}

TEST_CASE("TICLCandidate returns the stored probabilities in the order of ParticleType", "[TICLCandidate]") {
  using T = io_v1::TICLCandidate::ParticleType;
  io_v1::TICLCandidate cand;
  const std::array<float, 8> id{0.1f, 0.2f, 0.3f, 0.f, 0.f, 0.4f, 0.f, 0.f};
  cand.setIdProbabilities(id);
  cand.setTypeProbabilities({0.f, 0.9f, 0.f, 0.f, 0.f, 0.05f, 0.f, 0.f});
  const float maxError = 0.5f / 65535.f + 1e-7f;
  const auto back = cand.idProbabilities();
  for (unsigned int i = 0; i < id.size(); ++i)
    REQUIRE(std::abs(back[i] - id[i]) <= maxError);
  REQUIRE(std::abs(cand.id_probability(T::photon) - 0.1f) <= maxError);
  REQUIRE(std::abs(cand.typeProbability(T::electron) - 0.9f) <= maxError);
  REQUIRE(cand.typeProbabilities()[0] == 0.f);

  cand.setIdProbability(T::muon, 1.f);
  REQUIRE(cand.id_probability(T::muon) == 1.f);
  cand.zeroProbabilities();
  for (const float p : cand.idProbabilities())
    REQUIRE(p == 0.f);
}

TEST_CASE("TICLCandidate has no time, no MTD time and zero probabilities by default", "[TICLCandidate]") {
  for (const auto& cand : {io_v1::TICLCandidate(), io_v1::TICLCandidate(1, reco::Candidate::LorentzVector())}) {
    REQUIRE(cand.time() == 0.f);
    REQUIRE(cand.timeError() == -1.f);
    REQUIRE(cand.MTDtime() == 0.f);
    REQUIRE(cand.MTDtimeError() == -1.f);
    REQUIRE(cand.rawEnergy() == 0.f);
    for (const float p : cand.idProbabilities())
      REQUIRE(p == 0.f);
    for (const float p : cand.typeProbabilities())
      REQUIRE(p == 0.f);
  }
}

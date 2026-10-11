#ifndef RecoMuon_L3TrackFinder_MuonChamberMatch_h
#define RecoMuon_L3TrackFinder_MuonChamberMatch_h

// Matching of a tracker track to the muon chambers (segments and rec-hits) for the muon high-purity
// track selection. It gives the raw quantities of the candidates in each muon station.
//
// The magnetic field bends a muon of charge q and transverse momentum pT. Thus its segment or
// rec-hits in each station are displaced in phi from the track direction (eta, phi) at the point
// of closest approach. For a pair of a track and an object:
//
//   bend    = q * dphi(position)  * pT   [rad GeV]
//   dirBend = q * dphi(direction) * pT   (segments only)
//   dEta    = eta(object) - eta(track)
//
// bend and dirBend are negative for an object displaced to the side to which the charge bends.
// The factor pT removes most of the momentum dependence. The values for the muon's own objects
// depend on the detector, the station and |eta|. The code does not use a table or a fit for them:
// the forest gets the raw quantities.
//
// The candidates of a track in a station are the objects in the search window: |dEta| <
// kChamberMaxDEta, bend on the charge side and not more than kChamberMaxBend, rec-hits within
// kChamberMaxHitBx bunch crossings, and for segments a direction that also bends to the charge
// side. The candidates are sorted by |dEta|, because the field does not bend eta, and the first
// two are kept. The window is wide: it is not a working point.
//
// The training extraction must use the same window, the same candidate order, inputs rounded to
// float and arithmetic in double. If this code changes, the forest must be trained again.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace muonhp {

  // Muon stations: ME0 segments, DT or CSC segments of stations 1 to 4, GE1/1 and GE2/1 rec-hits (the HLT has no
  // segments there), and RPC rec-hits of stations 1 to 4 (barrel or endcap).
  enum ChamberStation : uint8_t {
    kME0 = 0,
    kSt1,
    kSt2,
    kSt3,
    kSt4,
    kGE11,
    kGE21,
    kRPC1,
    kRPC2,
    kRPC3,
    kRPC4,
    kNChamberStations
  };

  // Search window
  inline constexpr double kChamberMaxDEta = 0.3;
  inline constexpr double kChamberMaxBend = 8.0;  // |q * dphi * pT| [rad GeV]
  inline constexpr int kChamberMaxHitBx = 1;      // rec-hits: |bx| <= 1

  // Values for a station without a candidate. They are outside the range of all candidates (bend < 0,
  // bendDiff < kChamberMaxBend, |dEta| < kChamberMaxDEta), so their exact value has no effect on the forest.
  inline constexpr float kNoCandidateBend = 9.f;
  inline constexpr float kNoCandidateDEta = 0.5f;

  // A segment or rec-hit in global coordinates, with values rounded to float.
  struct ChamberPoint {
    float eta = 0.f;
    float phi = 0.f;
    float dirPhi = 0.f;  // segments only
    uint8_t station = kME0;
    bool isSegment = false;
  };

  // Transverse radius and pseudorapidity of a point from its float coordinates, with each operation rounded to
  // float. The training input computes x * x + y * y with two roundings. The compiler can fuse it into one
  // multiply-add with one rounding. The volatile products prevent this, so the values agree with the training
  // input to the last bit.
  inline float chamberPerp(float x, float y) {
    volatile float xx = x * x;
    volatile float yy = y * y;
    return std::sqrt(xx + yy);
  }
  inline float chamberEta(float x, float y, float z) { return ::asinhf(z / chamberPerp(x, y)); }

  // phi(a) - phi(b). Reduced to [-pi, pi] only when it is outside this range, as in the training extraction.
  inline double chamberDeltaPhi(double a, double b) {
    const double d = a - b;
    return std::abs(d) <= M_PI ? d : d - std::rint(d * (1.0 / (2.0 * M_PI))) * (2.0 * M_PI);
  }

  // Track parameters rounded to float, given as double.
  struct ChamberMatchTrack {
    double pt, eta, phi, charge;
  };

  struct ChamberCandidate {
    float bend = kNoCandidateBend;
    float dEta = kNoCandidateDEta;
    float bendDiff =
        kNoCandidateBend;  // dirBend - bend (segments): how much more the direction bends than the position
  };

  // The two candidates nearest in eta of one station.
  struct StationCandidates {
    ChamberCandidate nearest;
    ChamberCandidate second;
  };

  using ChamberMatch = std::array<StationCandidates, kNChamberStations>;

  // points: the segments and rec-hits of the event in the order DT, CSC, ME0, GEM, RPC barrel, RPC endcap, each in
  // collection order. The training input uses the same order. Of two candidates with the same |dEta|, the first
  // one is the nearer.
  inline ChamberMatch matchChambers(const ChamberMatchTrack& t, const std::vector<ChamberPoint>& points) {
    ChamberMatch match;
    std::array<double, kNChamberStations> nearest, second;
    nearest.fill(std::numeric_limits<double>::max());
    second.fill(std::numeric_limits<double>::max());
    for (const auto& p : points) {
      const double dEta = static_cast<double>(p.eta) - t.eta;
      const double absDEta = std::abs(dEta);
      if (!(absDEta < kChamberMaxDEta)) {
        continue;
      }
      const double bend = t.charge * chamberDeltaPhi(p.phi, t.phi) * t.pt;
      if (!(bend < 0. && bend > -kChamberMaxBend)) {
        continue;
      }
      double dirBend = 0.;
      if (p.isSegment) {
        dirBend = t.charge * chamberDeltaPhi(p.dirPhi, t.phi) * t.pt;
        if (!(dirBend < 0.)) {
          continue;
        }
      }
      const ChamberCandidate candidate{
          static_cast<float>(bend), static_cast<float>(dEta), static_cast<float>(dirBend - bend)};
      auto& station = match[p.station];
      if (absDEta < nearest[p.station]) {
        station.second = station.nearest;
        second[p.station] = nearest[p.station];
        station.nearest = candidate;
        nearest[p.station] = absDEta;
      } else if (absDEta < second[p.station]) {
        station.second = candidate;
        second[p.station] = absDEta;
      }
    }
    return match;
  }

}  // namespace muonhp

#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include "RecoTracker/LSTGeometry/interface/Helix.h"
#include "RecoTracker/LSTGeometry/interface/ModuleMap.h"

namespace lstgeometry {

  struct EtaPhiBounds {
    float minEta = std::numeric_limits<float>::max();
    float maxEta = std::numeric_limits<float>::lowest();
    float minPhi = std::numeric_limits<float>::max();
    float maxPhi = std::numeric_limits<float>::lowest();
  };

  struct CornerCoordinates {
    // Columns are z, 1/r, phi for each module corner.
    MatrixF4x3 values;
    // Columns are eta at z shifts 0, +10, -10.
    MatrixF4x3 shiftedEtas;
    float centerPhi;
  };

  struct EtaPhiQuad {
    MatrixF4x2 corners;
    EtaPhiBounds bounds;
  };

  struct EtaPhiPoint {
    float eta;
    float phi;
  };

  struct RelativePhiData {
    std::array<float, 4> corners;
    float minPhi;
    float maxPhi;
  };

  using CornerCoordinatesMap = std::unordered_map<unsigned int, CornerCoordinates>;

  struct ModuleCandidate {
    unsigned int detid;
    CornerCoordinates const* corners;
  };

  struct MatchedCandidate {
    ModuleCandidate const* candidate;
    RelativePhiData relativePhis;
  };

  using BinnedCandidates = std::array<
      std::array<std::array<std::array<std::vector<ModuleCandidate>, kNPhiBins>, kNThetaBins>, kBarrelLayers + 1>,
      2>;

  std::vector<ModuleCandidate>& candidatesAt(
      BinnedCandidates& candidates, Location location, unsigned int layer, unsigned int thetaBin, unsigned int phiBin) {
    return candidates[locationIndex(location)][layer][thetaBin][phiBin];
  }

  std::vector<ModuleCandidate> const& candidatesAt(BinnedCandidates const& candidates,
                                                   Location location,
                                                   unsigned int layer,
                                                   unsigned int thetaBin,
                                                   unsigned int phiBin) {
    return candidates[locationIndex(location)][layer][thetaBin][phiBin];
  }

  BinnedCandidates buildBinnedCandidates(BinnedDetIds const& binned_detids,
                                         CornerCoordinatesMap const& corner_coordinates) {
    BinnedCandidates candidates;
    for (Location location : {Location::barrel, Location::endcap}) {
      for (unsigned int layer = 1; layer <= kBarrelLayers; ++layer) {
        for (unsigned int thetaBin = 0; thetaBin < kNThetaBins; ++thetaBin) {
          for (unsigned int phiBin = 0; phiBin < kNPhiBins; ++phiBin) {
            auto const& detids = binnedDetIdsAt(binned_detids, location, layer, thetaBin, phiBin);
            auto& binCandidates = candidatesAt(candidates, location, layer, thetaBin, phiBin);
            binCandidates.reserve(detids.size());
            for (unsigned int detid : detids)
              binCandidates.push_back({detid, &corner_coordinates.at(detid)});
          }
        }
      }
    }
    return candidates;
  }

  float normalizePhi(float phi) {
    constexpr float pi = std::numbers::pi_v<float>;
    constexpr float twoPi = 2.f * pi;
    if (phi >= pi)
      phi -= twoPi;
    else if (phi < -pi)
      phi += twoPi;
    return phi;
  }

  CornerCoordinates getCornerCoordinates(Sensor const& sensor) {
    CornerCoordinates coordinates;
    coordinates.centerPhi = sensor.centerPhi;
    auto const& corners = sensor.extra->corners;
    for (int i = 0; i < 4; ++i) {
      float x = corners(i, 1);
      float y = corners(i, 2);
      coordinates.values(i, 0) = corners(i, 0);
      coordinates.values(i, 1) = 1.f / std::sqrt(x * x + y * y);
      coordinates.values(i, 2) = std::atan2(y, x);
      coordinates.shiftedEtas(i, 0) = std::asinh(coordinates.values(i, 0) * coordinates.values(i, 1));
      coordinates.shiftedEtas(i, 1) = std::asinh((coordinates.values(i, 0) + 10.f) * coordinates.values(i, 1));
      coordinates.shiftedEtas(i, 2) = std::asinh((coordinates.values(i, 0) - 10.f) * coordinates.values(i, 1));
    }
    return coordinates;
  }

  bool etaPhiBoundsOverlap(EtaPhiBounds const& lhs, EtaPhiBounds const& rhs) {
    return lhs.minEta <= rhs.maxEta && lhs.maxEta >= rhs.minEta && lhs.minPhi <= rhs.maxPhi && lhs.maxPhi >= rhs.minPhi;
  }

  bool etaPhiBoundsContain(EtaPhiBounds const& bounds, EtaPhiPoint const& point) {
    return point.eta >= bounds.minEta && point.eta <= bounds.maxEta && point.phi >= bounds.minPhi &&
           point.phi <= bounds.maxPhi;
  }

  float cross2D(MatrixF4x2 const& points, int a, int b, int c) {
    float abEta = points(b, 0) - points(a, 0);
    float abPhi = points(b, 1) - points(a, 1);
    float acEta = points(c, 0) - points(a, 0);
    float acPhi = points(c, 1) - points(a, 1);
    return abEta * acPhi - abPhi * acEta;
  }

  float cross2D(MatrixF4x2 const& segment, int a, int b, MatrixF4x2 const& points, int c) {
    float abEta = segment(b, 0) - segment(a, 0);
    float abPhi = segment(b, 1) - segment(a, 1);
    float acEta = points(c, 0) - segment(a, 0);
    float acPhi = points(c, 1) - segment(a, 1);
    return abEta * acPhi - abPhi * acEta;
  }

  bool pointOnSegment(MatrixF4x2 const& segment, int a, int b, MatrixF4x2 const& points, int c) {
    constexpr float eps = 1e-6;
    if (std::fabs(cross2D(segment, a, b, points, c)) > eps)
      return false;
    return points(c, 0) >= std::min(segment(a, 0), segment(b, 0)) - eps &&
           points(c, 0) <= std::max(segment(a, 0), segment(b, 0)) + eps &&
           points(c, 1) >= std::min(segment(a, 1), segment(b, 1)) - eps &&
           points(c, 1) <= std::max(segment(a, 1), segment(b, 1)) + eps;
  }

  bool segmentsIntersect(MatrixF4x2 const& lhs, int lhsA, int lhsB, MatrixF4x2 const& rhs, int rhsA, int rhsB) {
    constexpr float eps = 1e-6;
    float lhsCrossA = cross2D(lhs, lhsA, lhsB, rhs, rhsA);
    float lhsCrossB = cross2D(lhs, lhsA, lhsB, rhs, rhsB);
    float rhsCrossA = cross2D(rhs, rhsA, rhsB, lhs, lhsA);
    float rhsCrossB = cross2D(rhs, rhsA, rhsB, lhs, lhsB);

    if (std::fabs(lhsCrossA) <= eps && pointOnSegment(lhs, lhsA, lhsB, rhs, rhsA))
      return true;
    if (std::fabs(lhsCrossB) <= eps && pointOnSegment(lhs, lhsA, lhsB, rhs, rhsB))
      return true;
    if (std::fabs(rhsCrossA) <= eps && pointOnSegment(rhs, rhsA, rhsB, lhs, lhsA))
      return true;
    if (std::fabs(rhsCrossB) <= eps && pointOnSegment(rhs, rhsA, rhsB, lhs, lhsB))
      return true;

    return (lhsCrossA > eps) != (lhsCrossB > eps) && (rhsCrossA > eps) != (rhsCrossB > eps);
  }

  bool pointInPolygon(MatrixF4x2 const& polygon, MatrixF4x2 const& points, int point) {
    bool inside = false;
    float pointEta = points(point, 0);
    float pointPhi = points(point, 1);

    for (int i = 0, j = 3; i < 4; j = i++) {
      if (pointOnSegment(polygon, j, i, points, point))
        return true;

      bool crossesPhi = (polygon(i, 1) > pointPhi) != (polygon(j, 1) > pointPhi);
      if (!crossesPhi)
        continue;

      float etaAtPointPhi =
          (polygon(j, 0) - polygon(i, 0)) * (pointPhi - polygon(i, 1)) / (polygon(j, 1) - polygon(i, 1)) +
          polygon(i, 0);
      if (pointEta < etaAtPointPhi)
        inside = !inside;
    }

    return inside;
  }

  bool pointInQuad(EtaPhiQuad const& quad, EtaPhiPoint const& point) {
    if (!etaPhiBoundsContain(quad.bounds, point))
      return false;

    bool inside = false;
    for (int i = 0, j = 3; i < 4; j = i++) {
      EtaPhiPoint start{quad.corners(j, 0), quad.corners(j, 1)};
      EtaPhiPoint end{quad.corners(i, 0), quad.corners(i, 1)};
      float cross = (end.eta - start.eta) * (point.phi - start.phi) - (end.phi - start.phi) * (point.eta - start.eta);
      constexpr float eps = 1e-6;
      if (std::fabs(cross) <= eps && point.eta >= std::min(start.eta, end.eta) - eps &&
          point.eta <= std::max(start.eta, end.eta) + eps && point.phi >= std::min(start.phi, end.phi) - eps &&
          point.phi <= std::max(start.phi, end.phi) + eps) {
        return true;
      }

      bool crossesPhi = (end.phi > point.phi) != (start.phi > point.phi);
      if (!crossesPhi)
        continue;

      float etaAtPointPhi = (start.eta - end.eta) * (point.phi - end.phi) / (start.phi - end.phi) + end.eta;
      if (point.eta < etaAtPointPhi)
        inside = !inside;
    }

    return inside;
  }

  bool quadIntersects(MatrixF4x2 const& lhs, MatrixF4x2 const& rhs) {
    for (int lhsEdge = 0; lhsEdge < 4; ++lhsEdge) {
      int lhsNext = (lhsEdge + 1) % 4;
      for (int rhsEdge = 0; rhsEdge < 4; ++rhsEdge) {
        int rhsNext = (rhsEdge + 1) % 4;
        if (segmentsIntersect(lhs, lhsEdge, lhsNext, rhs, rhsEdge, rhsNext))
          return true;
      }
    }

    return pointInPolygon(lhs, rhs, 0) || pointInPolygon(rhs, lhs, 0);
  }

  EtaPhiQuad getEtaPhiQuad(MatrixF4x3 const& corners, float refphi, float zshift = 0) {
    EtaPhiQuad data;
    for (int i = 0; i < 4; ++i) {
      float x = corners(i, 1);
      float y = corners(i, 2);
      data.corners(i, 0) = std::asinh((corners(i, 0) + zshift) / std::sqrt(x * x + y * y));
      data.corners(i, 1) = normalizePhi(std::atan2(y, x) - refphi);
      data.bounds.minEta = std::min(data.bounds.minEta, data.corners(i, 0));
      data.bounds.maxEta = std::max(data.bounds.maxEta, data.corners(i, 0));
      data.bounds.minPhi = std::min(data.bounds.minPhi, data.corners(i, 1));
      data.bounds.maxPhi = std::max(data.bounds.maxPhi, data.corners(i, 1));
    }
    return data;
  }

  EtaPhiQuad getEtaPhiQuad(CornerCoordinates const& corners, float refphi, int etaColumn) {
    EtaPhiQuad data;
    for (int i = 0; i < 4; ++i) {
      data.corners(i, 0) = corners.shiftedEtas(i, etaColumn);
      data.corners(i, 1) = normalizePhi(corners.values(i, 2) - refphi);
      data.bounds.minEta = std::min(data.bounds.minEta, data.corners(i, 0));
      data.bounds.maxEta = std::max(data.bounds.maxEta, data.corners(i, 0));
      data.bounds.minPhi = std::min(data.bounds.minPhi, data.corners(i, 1));
      data.bounds.maxPhi = std::max(data.bounds.maxPhi, data.corners(i, 1));
    }
    return data;
  }

  RelativePhiData getRelativePhiData(CornerCoordinates const& corners, float refphi) {
    RelativePhiData data;
    data.corners[0] = normalizePhi(corners.values(0, 2) - refphi);
    data.minPhi = data.maxPhi = data.corners[0];
    for (int i = 1; i < 4; ++i) {
      data.corners[i] = normalizePhi(corners.values(i, 2) - refphi);
      data.minPhi = std::min(data.minPhi, data.corners[i]);
      data.maxPhi = std::max(data.maxPhi, data.corners[i]);
    }
    return data;
  }

  EtaPhiQuad getEtaPhiQuad(CornerCoordinates const& corners, RelativePhiData const& relativePhis, int etaColumn) {
    EtaPhiQuad data;
    data.bounds.minPhi = relativePhis.minPhi;
    data.bounds.maxPhi = relativePhis.maxPhi;
    for (int i = 0; i < 4; ++i) {
      data.corners(i, 0) = corners.shiftedEtas(i, etaColumn);
      data.corners(i, 1) = relativePhis.corners[i];
      data.bounds.minEta = std::min(data.bounds.minEta, data.corners(i, 0));
      data.bounds.maxEta = std::max(data.bounds.maxEta, data.corners(i, 0));
    }
    return data;
  }

  float getCenterPhi(MatrixF4x3 const& corners) {
    RowVectorF3 center = corners.colwise().sum();
    center /= 4.;
    return std::atan2(center(2), center(1));
  }

  bool moduleOverlapsInEtaPhi(EtaPhiQuad const& ref_mod_boundaries_etaphi,
                              CornerCoordinates const& tar_mod_boundaries,
                              RelativePhiData const& relativePhis,
                              int etaColumn) {
    MatrixF4x2 tar_mod_boundaries_etaphi;
    EtaPhiBounds tar_bounds;
    tar_mod_boundaries_etaphi(0, 0) = tar_mod_boundaries.shiftedEtas(0, etaColumn);
    tar_mod_boundaries_etaphi(0, 1) = relativePhis.corners[0];
    tar_bounds.minEta = tar_bounds.maxEta = tar_mod_boundaries_etaphi(0, 0);
    tar_bounds.minPhi = relativePhis.minPhi;
    tar_bounds.maxPhi = relativePhis.maxPhi;

    // Quick cut
    if (std::fabs(ref_mod_boundaries_etaphi.corners(0, 0) - tar_mod_boundaries_etaphi(0, 0)) > 0.5)
      return false;
    if (std::fabs(normalizePhi(ref_mod_boundaries_etaphi.corners(0, 1) - tar_mod_boundaries_etaphi(0, 1))) > 1.)
      return false;

    for (int i = 1; i < 4; ++i) {
      tar_mod_boundaries_etaphi(i, 0) = tar_mod_boundaries.shiftedEtas(i, etaColumn);
      tar_mod_boundaries_etaphi(i, 1) = relativePhis.corners[i];
      tar_bounds.minEta = std::min(tar_bounds.minEta, tar_mod_boundaries_etaphi(i, 0));
      tar_bounds.maxEta = std::max(tar_bounds.maxEta, tar_mod_boundaries_etaphi(i, 0));
    }

    if (!etaPhiBoundsOverlap(ref_mod_boundaries_etaphi.bounds, tar_bounds))
      return false;

    return quadIntersects(ref_mod_boundaries_etaphi.corners, tar_mod_boundaries_etaphi);
  }

  float quadArea(EtaPhiQuad const& quad) {
    float area = 0.;
    for (int i = 0; i < 4; ++i) {
      int next = (i + 1) % 4;
      area += quad.corners(i, 0) * quad.corners(next, 1) - quad.corners(next, 0) * quad.corners(i, 1);
    }
    return 0.5f * std::fabs(area);
  }

  bool pointCovered(EtaPhiPoint const& point, std::vector<EtaPhiQuad> const& covering_quads) {
    for (auto const& quad : covering_quads) {
      if (pointInQuad(quad, point))
        return true;
    }
    return false;
  }

  float approximateUncoveredArea(EtaPhiQuad const& ref_quad,
                                 std::vector<EtaPhiQuad> const& covering_quads,
                                 std::vector<EtaPhiPoint>& uncovered_points) {
    constexpr int kGrid = 8;
    unsigned int nInside = 0;
    unsigned int nUncovered = 0;
    uncovered_points.clear();
    uncovered_points.reserve(kGrid * kGrid);

    for (int etaBin = 0; etaBin < kGrid; ++etaBin) {
      float eta = ref_quad.bounds.minEta +
                  (etaBin + 0.5f) * (ref_quad.bounds.maxEta - ref_quad.bounds.minEta) / static_cast<float>(kGrid);
      for (int phiBin = 0; phiBin < kGrid; ++phiBin) {
        float phi = ref_quad.bounds.minPhi +
                    (phiBin + 0.5f) * (ref_quad.bounds.maxPhi - ref_quad.bounds.minPhi) / static_cast<float>(kGrid);
        EtaPhiPoint point{eta, phi};
        if (!pointInQuad(ref_quad, point))
          continue;

        ++nInside;
        if (!pointCovered(point, covering_quads)) {
          ++nUncovered;
          uncovered_points.push_back(point);
        }
      }
    }

    if (nInside == 0)
      return 0.;
    return quadArea(ref_quad) * static_cast<float>(nUncovered) / static_cast<float>(nInside);
  }

  bool targetIntersectsApproxUncovered(EtaPhiQuad const& ref_quad,
                                       std::vector<EtaPhiQuad> const& covering_quads,
                                       std::vector<EtaPhiPoint> const& uncovered_points,
                                       EtaPhiQuad const& target_quad) {
    if (!etaPhiBoundsOverlap(ref_quad.bounds, target_quad.bounds))
      return false;

    for (auto const& point : uncovered_points) {
      if (pointInQuad(target_quad, point))
        return true;
    }

    for (int i = 0; i < 4; ++i) {
      EtaPhiPoint target_corner{target_quad.corners(i, 0), target_quad.corners(i, 1)};
      if (pointInQuad(ref_quad, target_corner) && !pointCovered(target_corner, covering_quads))
        return true;
    }

    return false;
  }

  std::vector<unsigned int> getStraightLineConnections(Sensor const& ref_sensor,
                                                       CornerCoordinates const& ref_corners,
                                                       BinnedCandidates const& binned_candidates) {
    float refphi = ref_sensor.centerPhi;
    unsigned short ref_layer = ref_sensor.extra->layer;
    auto ref_location = ref_sensor.extra->location;

    auto thetaphibins = getThetaPhiBins(ref_sensor.extra->centerTheta, ref_sensor.centerPhi);

    auto const& tar_detids_to_be_considered =
        candidatesAt(binned_candidates, ref_location, ref_layer + 1, thetaphibins.first, thetaphibins.second);

    constexpr std::array<int, 3> etaColumns = {0, 1, 2};
    std::array<EtaPhiQuad, 3> ref_quads = {getEtaPhiQuad(ref_corners, refphi, etaColumns[0]),
                                           getEtaPhiQuad(ref_corners, refphi, etaColumns[1]),
                                           getEtaPhiQuad(ref_corners, refphi, etaColumns[2])};

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    list_of_detids_etaphi_layer_tar.reserve(tar_detids_to_be_considered.size());
    std::vector<MatchedCandidate> list_of_candidates_etaphi_layer_tar;
    list_of_candidates_etaphi_layer_tar.reserve(tar_detids_to_be_considered.size());
    for (auto const& candidate : tar_detids_to_be_considered) {
      auto const& tar_corners = *candidate.corners;
      if (std::fabs(normalizePhi(ref_sensor.centerPhi - tar_corners.centerPhi)) > std::numbers::pi_v<float> / 2.)
        continue;
      RelativePhiData relativePhis = getRelativePhiData(tar_corners, refphi);
      for (unsigned int i = 0; i < etaColumns.size(); ++i) {
        if (moduleOverlapsInEtaPhi(ref_quads[i], tar_corners, relativePhis, etaColumns[i])) {
          list_of_detids_etaphi_layer_tar.push_back(candidate.detid);
          list_of_candidates_etaphi_layer_tar.push_back({&candidate, relativePhis});
          break;
        }
      }
    }

    // Consider barrel to endcap connections if the approximated uncovered area is > 0
    // after accounting for target modules in the next barrel layer.
    if (ref_location == Location::barrel) {
      std::vector<unsigned int> barrel_endcap_connected_tar_detids;
      std::vector<EtaPhiQuad> covering_quads;
      std::vector<EtaPhiPoint> uncovered_points;
      covering_quads.reserve(list_of_candidates_etaphi_layer_tar.size());
      uncovered_points.reserve(64);

      for (unsigned int i = 0; i < etaColumns.size(); ++i) {
        // Check whether there is still significant non-zero area
        covering_quads.clear();
        for (auto const& candidate : list_of_candidates_etaphi_layer_tar) {
          EtaPhiQuad tar_quad = getEtaPhiQuad(*candidate.candidate->corners, candidate.relativePhis, etaColumns[i]);
          if (etaPhiBoundsOverlap(ref_quads[i].bounds, tar_quad.bounds))
            covering_quads.push_back(std::move(tar_quad));
        }
        float area = approximateUncoveredArea(ref_quads[i], covering_quads, uncovered_points);

        if (area <= 5e-3)
          continue;

        auto const& new_tar_detids_to_be_considered =
            candidatesAt(binned_candidates, Location::endcap, 1, thetaphibins.first, thetaphibins.second);

        for (auto const& candidate : new_tar_detids_to_be_considered) {
          auto const& tar_corners = *candidate.corners;
          float tarphi = tar_corners.centerPhi;

          if (std::fabs(normalizePhi(tarphi - refphi)) > std::numbers::pi_v<float> / 2.)
            continue;

          RelativePhiData relativePhis = getRelativePhiData(tar_corners, refphi);
          EtaPhiQuad target_quad = getEtaPhiQuad(tar_corners, relativePhis, etaColumns[i]);
          if (targetIntersectsApproxUncovered(ref_quads[i], covering_quads, uncovered_points, target_quad))
            barrel_endcap_connected_tar_detids.push_back(candidate.detid);
        }
      }
      list_of_detids_etaphi_layer_tar.insert(list_of_detids_etaphi_layer_tar.end(),
                                             barrel_endcap_connected_tar_detids.begin(),
                                             barrel_endcap_connected_tar_detids.end());
    }

    return list_of_detids_etaphi_layer_tar;
  }

  MatrixF4x3 boundsAfterCurved(Sensor const& ref_sensor,
                               std::array<float, kBarrelLayers> const& average_r_barrel,
                               std::array<float, kEndcapLayers> const& average_z_endcap,
                               float ptCut) {
    auto const& bounds = ref_sensor.extra->corners;
    int charge = 1;
    float z_r = ref_sensor.centerZ /
                std::sqrt(ref_sensor.centerX * ref_sensor.centerX + ref_sensor.centerY * ref_sensor.centerY);
    float refphi = ref_sensor.centerPhi;
    unsigned short ref_layer = ref_sensor.extra->layer;
    auto ref_location = ref_sensor.extra->location;
    MatrixF4x3 next_layer_bound_points;

    for (int i = 0; i < bounds.rows(); i++) {
      float bound_z_r = bounds(i, 0) / std::sqrt(bounds(i, 1) * bounds(i, 1) + bounds(i, 2) * bounds(i, 2));
      float bound_phi = std::atan2(bounds(i, 2), bounds(i, 1));
      float phi_diff = normalizePhi(bound_phi - refphi);
      int helixCharge = phi_diff > 0 ? -charge : charge;
      float vertexZ = bound_z_r < z_r ? 10.f : -10.f;
      Helix helix(ptCut, 0, 0, vertexZ, bounds(i, 1), bounds(i, 2), bounds(i, 0), helixCharge);

      std::tuple<float, float, float, float> next_point;
      if (ref_location == Location::barrel) {
        float tar_layer_radius = average_r_barrel[ref_layer];
        next_point = helix.pointFromRadius(tar_layer_radius);
      } else {
        float tar_layer_z = average_z_endcap[ref_layer];
        next_point = helix.pointFromZ(std::copysign(tar_layer_z, helix.lambda));
      }
      next_layer_bound_points(i, 0) = std::get<2>(next_point);
      next_layer_bound_points(i, 1) = std::get<0>(next_point);
      next_layer_bound_points(i, 2) = std::get<1>(next_point);
    }

    return next_layer_bound_points;
  }

  std::vector<unsigned int> getCurvedLineConnections(Sensor const& ref_sensor,
                                                     BinnedCandidates const& binned_candidates,
                                                     std::array<float, kBarrelLayers> const& average_r_barrel,
                                                     std::array<float, kEndcapLayers> const& average_z_endcap,
                                                     float ptCut) {
    float refphi = ref_sensor.centerPhi;

    unsigned short ref_layer = ref_sensor.extra->layer;
    auto ref_location = ref_sensor.extra->location;

    auto thetaphibins = getThetaPhiBins(ref_sensor.extra->centerTheta, ref_sensor.centerPhi);

    auto const& tar_detids_to_be_considered =
        candidatesAt(binned_candidates, ref_location, ref_layer + 1, thetaphibins.first, thetaphibins.second);

    auto next_layer_bound_points = boundsAfterCurved(ref_sensor, average_r_barrel, average_z_endcap, ptCut);
    EtaPhiQuad next_layer_quad = getEtaPhiQuad(next_layer_bound_points, refphi);
    float next_layer_center_phi = getCenterPhi(next_layer_bound_points);

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    list_of_detids_etaphi_layer_tar.reserve(tar_detids_to_be_considered.size());
    std::vector<MatchedCandidate> list_of_candidates_etaphi_layer_tar;
    list_of_candidates_etaphi_layer_tar.reserve(tar_detids_to_be_considered.size());
    for (auto const& candidate : tar_detids_to_be_considered) {
      auto const& tar_corners = *candidate.corners;
      if (std::fabs(normalizePhi(next_layer_center_phi - tar_corners.centerPhi)) > std::numbers::pi_v<float> / 2.)
        continue;
      RelativePhiData relativePhis = getRelativePhiData(tar_corners, refphi);
      if (moduleOverlapsInEtaPhi(next_layer_quad, tar_corners, relativePhis, 0)) {
        list_of_detids_etaphi_layer_tar.push_back(candidate.detid);
        list_of_candidates_etaphi_layer_tar.push_back({&candidate, relativePhis});
      }
    }

    // Consider barrel to endcap connections if the approximated uncovered area is > 0
    // after accounting for target modules in the next barrel layer.
    if (ref_location == Location::barrel) {
      std::vector<unsigned int> barrel_endcap_connected_tar_detids;

      // Check whether there is still significant non-zero area
      std::vector<EtaPhiQuad> covering_quads;
      covering_quads.reserve(list_of_detids_etaphi_layer_tar.size());
      for (auto const& candidate : list_of_candidates_etaphi_layer_tar) {
        EtaPhiQuad tar_quad = getEtaPhiQuad(*candidate.candidate->corners, candidate.relativePhis, 0);
        if (etaPhiBoundsOverlap(next_layer_quad.bounds, tar_quad.bounds))
          covering_quads.push_back(std::move(tar_quad));
      }
      std::vector<EtaPhiPoint> uncovered_points;
      float area = approximateUncoveredArea(next_layer_quad, covering_quads, uncovered_points);

      if (area > 5e-3) {
        auto const& new_tar_detids_to_be_considered =
            candidatesAt(binned_candidates, Location::endcap, 1, thetaphibins.first, thetaphibins.second);

        for (auto const& candidate : new_tar_detids_to_be_considered) {
          auto const& tar_corners = *candidate.corners;
          float tarphi = tar_corners.centerPhi;

          if (std::fabs(normalizePhi(tarphi - refphi)) > std::numbers::pi_v<float> / 2.)
            continue;

          RelativePhiData relativePhis = getRelativePhiData(tar_corners, refphi);
          EtaPhiQuad target_quad = getEtaPhiQuad(tar_corners, relativePhis, 0);
          if (targetIntersectsApproxUncovered(next_layer_quad, covering_quads, uncovered_points, target_quad))
            barrel_endcap_connected_tar_detids.push_back(candidate.detid);
        }
      }

      list_of_detids_etaphi_layer_tar.insert(list_of_detids_etaphi_layer_tar.end(),
                                             barrel_endcap_connected_tar_detids.begin(),
                                             barrel_endcap_connected_tar_detids.end());
    }

    return list_of_detids_etaphi_layer_tar;
  }

  ModuleMap buildModuleMap(Sensors const& sensors,
                           BinnedDetIds const& binned_detids,
                           std::array<float, kBarrelLayers> const& average_r_barrel,
                           std::array<float, kEndcapLayers> const& average_z_endcap,
                           float pt_cut) {
    ModuleMap moduleMap;
    moduleMap.reserve(sensors.size());

    CornerCoordinatesMap corner_coordinates;
    corner_coordinates.reserve(sensors.size());
    for (auto const& [detid, sensor] : sensors) {
      if (!sensor.extra->lower)
        continue;
      corner_coordinates.emplace(detid, getCornerCoordinates(sensor));
    }
    BinnedCandidates binned_candidates = buildBinnedCandidates(binned_detids, corner_coordinates);

    for (auto const& [ref_detid, s] : sensors) {
      // exclude the outermost modules that do not have connections to other modules
      if (!((s.extra->location == Location::barrel && s.extra->lower && s.extra->layer != 6) ||
            (s.extra->location == Location::endcap && s.extra->lower && s.extra->layer != 5 &&
             !(s.extra->ring == 15 && s.extra->layer == 1) && !(s.extra->ring == 15 && s.extra->layer == 2) &&
             !(s.extra->ring == 12 && s.extra->layer == 3) && !(s.extra->ring == 12 && s.extra->layer == 4))))
        continue;
      auto const& ref_corners = corner_coordinates.at(ref_detid);
      auto straight_line_connections = getStraightLineConnections(s, ref_corners, binned_candidates);
      auto curved_line_connections =
          getCurvedLineConnections(s, binned_candidates, average_r_barrel, average_z_endcap, pt_cut);
      auto& connections = moduleMap[ref_detid];
      connections.reserve(straight_line_connections.size() + curved_line_connections.size());
      connections.insert(connections.end(), straight_line_connections.begin(), straight_line_connections.end());
      connections.insert(connections.end(), curved_line_connections.begin(), curved_line_connections.end());
      std::sort(connections.begin(), connections.end());
      connections.erase(std::unique(connections.begin(), connections.end()), connections.end());
    }

    return moduleMap;
  }

}  // namespace lstgeometry

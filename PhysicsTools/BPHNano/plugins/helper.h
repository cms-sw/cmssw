// original author: RK18 team

#ifndef PhysicsTools_BPHNano_helpers
#define PhysicsTools_BPHNano_helpers

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/PV3DBase.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "Math/LorentzVector.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "TVector3.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

typedef std::vector<reco::TransientTrack> TransientTrackCollection;

constexpr float PROT_MASS = 0.938272;
constexpr float K_MASS = 0.493677;
constexpr float PI_MASS = 0.139571;
constexpr float LEP_SIGMA = 0.0000001;
constexpr float K_SIGMA = 0.000016;
constexpr float PI_SIGMA = 0.000016;
constexpr float PROT_SIGMA = 0.000016;
constexpr float MUON_MASS = 0.10565837;
constexpr float ELECTRON_MASS = 0.000511;

inline std::pair<float, float> min_max_dr(
    const std::vector<edm::Ptr<reco::Candidate>>& cands) {
  float min_dr = std::numeric_limits<float>::max();
  float max_dr = 0.;
  for (size_t i = 0; i < cands.size(); ++i) {
    for (size_t j = i + 1; j < cands.size(); ++j) {
      float dr = reco::deltaR(*cands.at(i), *cands.at(j));
      min_dr = std::min(min_dr, dr);
      max_dr = std::max(max_dr, dr);
    }
  }
  return std::make_pair(min_dr, max_dr);
}

template <typename FITTER, typename LORENTZ_VEC>
inline double cos_theta_2D(const FITTER& fitter, const reco::BeamSpot& bs,
                           const LORENTZ_VEC& p4) {
  if (!fitter.success()) return -2;
  GlobalPoint point = fitter.fitted_vtx();
  auto bs_pos = bs.position(point.z());
  math::XYZVector delta(point.x() - bs_pos.x(), point.y() - bs_pos.y(), 0.);
  math::XYZVector pt(p4.px(), p4.py(), 0.);
  double den = (delta.R() * pt.R());
  return (den != 0.) ? delta.Dot(pt) / den : -2;
}

template <typename FITTER>
inline Measurement1D l_xy(const FITTER& fitter, const reco::BeamSpot& bs) {
  if (!fitter.success()) return {-2, -2};
  GlobalPoint point = fitter.fitted_vtx();
  GlobalError err = fitter.fitted_vtx_uncertainty();
  auto bs_pos = bs.position(point.z());
  GlobalPoint delta(point.x() - bs_pos.x(), point.y() - bs_pos.y(), 0.);
  return {delta.perp(), sqrt(err.rerr(delta))};
}

/*
inline GlobalPoint FlightDistVector (const reco::BeamSpot & bm, GlobalPoint
Bvtx)
{
   GlobalPoint Dispbeamspot(-1*( (bm.x0()-Bvtx.x()) + (Bvtx.z()-bm.z0()) *
bm.dxdz()), -1*( (bm.y0()-Bvtx.y()) + (Bvtx.z()-bm.z0()) * bm.dydz()), 0);
   return std::move(Dispbeamspot);
}
*/

inline float CosA(
    GlobalPoint& dist,
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>& Bp4) {
  math::XYZVector vperp(dist.x(), dist.y(), 0);
  math::XYZVector pperp(Bp4.Px(), Bp4.Py(), 0);
  return std::move(vperp.Dot(pperp) / (vperp.R() * pperp.R()));
}

inline std::pair<double, double> computeDCA(const reco::TransientTrack& trackTT,
                                            const reco::BeamSpot& beamSpot) {
  double DCABS = -1.;
  double DCABSErr = -1.;

  TrajectoryStateClosestToPoint theDCAXBS =
      trackTT.trajectoryStateClosestToPoint(
          GlobalPoint(beamSpot.position().x(), beamSpot.position().y(),
                      beamSpot.position().z()));
  if (theDCAXBS.isValid()) {
    DCABS = theDCAXBS.perigeeParameters().transverseImpactParameter();
    DCABSErr = theDCAXBS.perigeeError().transverseImpactParameterError();
  }

  return std::make_pair(DCABS, DCABSErr);
}

inline bool track_to_lepton_match(edm::Ptr<reco::Candidate> l_ptr,
                                  auto iso_tracks_id, unsigned int iTrk) {
  for (unsigned int i = 0; i < l_ptr->numberOfSourceCandidatePtrs(); ++i) {
    if (!((l_ptr->sourceCandidatePtr(i)).isNonnull() &&
          (l_ptr->sourceCandidatePtr(i)).isAvailable()))
      continue;
    const edm::Ptr<reco::Candidate>& source = l_ptr->sourceCandidatePtr(i);
    if (source.id() == iso_tracks_id && source.key() == iTrk) {
      return true;
    }
  }
  return false;
}

inline std::pair<bool, Measurement1D> absoluteImpactParameter(
    const TrajectoryStateOnSurface& tsos, RefCountedKinematicVertex vertex,
    VertexDistance& distanceComputer) {
  if (!tsos.isValid()) {
    return std::pair<bool, Measurement1D>(false, Measurement1D(0., 0.));
  }
  GlobalPoint refPoint = tsos.globalPosition();
  GlobalError refPointErr = tsos.cartesianError().position();
  GlobalPoint vertexPosition = vertex->vertexState().position();
  GlobalError vertexPositionErr =
      RecoVertex::convertError(vertex->vertexState().error());
  return std::pair<bool, Measurement1D>(
      true,
      distanceComputer.distance(VertexState(vertexPosition, vertexPositionErr),
                                VertexState(refPoint, refPointErr)));
}

inline std::pair<bool, Measurement1D> absoluteImpactParameter3D(
    const TrajectoryStateOnSurface& tsos, RefCountedKinematicVertex vertex) {
  VertexDistance3D dist;
  return absoluteImpactParameter(tsos, vertex, dist);
}

inline std::pair<bool, Measurement1D> absoluteTransverseImpactParameter(
    const TrajectoryStateOnSurface& tsos, RefCountedKinematicVertex vertex) {
  VertexDistanceXY dist;
  return absoluteImpactParameter(tsos, vertex, dist);
}

inline std::pair<bool, Measurement1D> signedImpactParameter3D(
    const TrajectoryStateOnSurface& tsos, RefCountedKinematicVertex vertex,
    const reco::BeamSpot& bs, double pv_z) {
  VertexDistance3D dist;

  std::pair<bool, Measurement1D> result =
      absoluteImpactParameter(tsos, vertex, dist);
  if (!result.first) return result;

  // Compute Sign
  auto bs_pos = bs.position(vertex->vertexState().position().z());
  GlobalPoint impactPoint = tsos.globalPosition();
  GlobalVector IPVec(impactPoint.x() - vertex->vertexState().position().x(),
                     impactPoint.y() - vertex->vertexState().position().y(),
                     impactPoint.z() - vertex->vertexState().position().z());

  GlobalVector direction(vertex->vertexState().position().x() - bs_pos.x(),
                         vertex->vertexState().position().y() - bs_pos.y(),
                         vertex->vertexState().position().z() - pv_z);

  double prod = IPVec.dot(direction);
  double sign = (prod >= 0) ? 1. : -1.;

  // Apply sign to the result
  return std::pair<bool, Measurement1D>(
      result.first,
      Measurement1D(sign * result.second.value(), result.second.error()));
}

inline std::pair<bool, Measurement1D> signedTransverseImpactParameter(
    const TrajectoryStateOnSurface& tsos, RefCountedKinematicVertex vertex,
    const reco::BeamSpot& bs) {
  VertexDistanceXY dist;

  std::pair<bool, Measurement1D> result =
      absoluteImpactParameter(tsos, vertex, dist);
  if (!result.first) return result;

  // Compute Sign
  auto bs_pos = bs.position(vertex->vertexState().position().z());
  GlobalPoint impactPoint = tsos.globalPosition();
  GlobalVector IPVec(impactPoint.x() - vertex->vertexState().position().x(),
                     impactPoint.y() - vertex->vertexState().position().y(),
                     0.);
  GlobalVector direction(vertex->vertexState().position().x() - bs_pos.x(),
                         vertex->vertexState().position().y() - bs_pos.y(), 0);

  double prod = IPVec.dot(direction);
  double sign = (prod >= 0) ? 1. : -1.;

  // Apply sign to the result
  return std::pair<bool, Measurement1D>(
      result.first,
      Measurement1D(sign * result.second.value(), result.second.error()));
}

inline std::vector<float> TrackerIsolation(
    edm::Handle<pat::CompositeCandidateCollection>& tracks,
    pat::CompositeCandidate& B, std::vector<std::string>& dnames) {
  std::vector<float> iso(dnames.size(), 0);
  for (size_t k_idx = 0; k_idx < tracks->size(); ++k_idx) {
    edm::Ptr<pat::CompositeCandidate> trk_ptr(tracks, k_idx);
    for (size_t iname = 0; iname < dnames.size(); ++iname) {
      float dr = deltaR(B.userFloat("fitted_" + dnames[iname] + "_eta"),
                        B.userFloat("fitted_" + dnames[iname] + "_phi"),
                        trk_ptr->eta(), trk_ptr->phi());
      if (dr > 0 && dr < 0.4) iso[iname] += trk_ptr->pt();
    }
  }
  return iso;
}

#endif

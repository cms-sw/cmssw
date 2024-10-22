#include "CommonTools/RecoAlgos/interface/PrimaryVertexAssignment.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "FWCore/Utilities/interface/isFinite.h"

std::pair<int, PrimaryVertexAssignment::Quality> PrimaryVertexAssignment::chargedHadronVertex(
    const reco::VertexCollection& vertices,
    const reco::TrackRef& trackRef,
    const reco::Track* track,
    float time,
    float timeReso,  // <0 if timing not available for this object
    const edm::View<reco::Candidate>& jets,
    const TransientTrackBuilder& builder) const {
  typedef reco::VertexCollection::const_iterator IV;
  typedef reco::Vertex::trackRef_iterator IT;

  int iVertex = -1;
  size_t index = 0;
  float bestweight = 0;
  for (auto const& vtx : vertices) {
    float w = vtx.trackWeight(trackRef);
    if (w > bestweight) {
      bestweight = w;
      iVertex = index;
    }
    index++;
  }
  return chargedHadronVertex(vertices, iVertex, track, time, timeReso, jets, builder);
}

std::pair<int, PrimaryVertexAssignment::Quality> PrimaryVertexAssignment::chargedHadronVertex(
    const reco::VertexCollection& vertices,
    int iVertex,
    const reco::Track* track,
    float time,
    float timeReso,  // <0 if timing not available for this object
    const edm::View<reco::Candidate>& jets,
    const TransientTrackBuilder& builder) const {
  typedef reco::VertexCollection::const_iterator IV;
  typedef reco::Vertex::trackRef_iterator IT;

  bool useTime = useTiming_;
  if (edm::isNotFinite(time) || timeReso < 1e-6) {
    useTime = false;
    time = 0.;
    timeReso = -1.;
  }

  if (preferHighRanked_) {
    for (IV iv = vertices.begin(); iv != vertices.end(); ++iv) {
      int ivtx = iv - vertices.begin();
      if (useVertexFit_ && (iVertex == ivtx))
        return {ivtx, PrimaryVertexAssignment::UsedInFit};

      double dz = std::abs(track->dz(iv->position()));
      double dt = std::abs(time - iv->t());

      bool useTimeVtx = useTime && iv->tError() > 0.;

      if ((dz < maxDzForPrimaryAssignment_ or dz / track->dzError() < maxDzSigForPrimaryAssignment_) and
          (!useTimeVtx or dt / timeReso < maxDtSigForPrimaryAssignment_)) {
        return {ivtx, PrimaryVertexAssignment::PrimaryDz};
      }
    }
  }

  double firstVertexDz = std::numeric_limits<double>::max();
  if (!vertices.empty())
    firstVertexDz = std::abs(track->dz(vertices[0].position()));
  // recover cases where the primary vertex is split
  if (useVertexFit_ && (iVertex > 0) && (iVertex <= int(fNumOfPUVtxsForCharged_)) &&
      firstVertexDz < fDzCutForChargedFromPUVtxs_)
    return {0, PrimaryVertexAssignment::PrimaryDz};

  if (useVertexFit_ && (iVertex >= 0))
    return {iVertex, PrimaryVertexAssignment::UsedInFit};

  double distmin = std::numeric_limits<double>::max();
  double dzmin = std::numeric_limits<double>::max();
  double dtmin = std::numeric_limits<double>::max();
  int vtxIdMinSignif = -1;
  for (IV iv = vertices.begin(); iv != vertices.end(); ++iv) {
    double dz = std::abs(track->dz(iv->position()));
    double dt = std::abs(time - iv->t());

    double dzsig = dz / track->dzError();
    double dist = dzsig * dzsig;

    bool useTimeVtx = useTime && iv->tError() > 0.;
    if (useTime && !useTimeVtx) {
      dt = timeReso;
    }

    if (useTime) {
      double dtsig = dt / timeReso;

      dist += dtsig * dtsig;
    }
    if (dist < distmin) {
      distmin = dist;
      dzmin = dz;
      dtmin = dt;
      vtxIdMinSignif = iv - vertices.begin();
    }
  }

  // protect high pT particles from association to pileup vertices and assign them to the first vertex
  if ((fPtMaxCharged_ > 0) && (vtxIdMinSignif >= 0) && (track->pt() > fPtMaxCharged_)) {
    iVertex = 0;
  } else {
    // first use "closest in Z" with tight cuts (targetting primary particles)
    const float add_cov = vtxIdMinSignif >= 0 ? vertices[vtxIdMinSignif].covariance(2, 2) : 0.f;
    const float dzE = sqrt(track->dzError() * track->dzError() + add_cov);
    if (!fOnlyUseFirstDz_) {
      if (vtxIdMinSignif >= 0 and
          (dzmin < maxDzForPrimaryAssignment_ and dzmin / dzE < maxDzSigForPrimaryAssignment_ and
           track->dzError() < maxDzErrorForPrimaryAssignment_) and
          (!useTime or dtmin / timeReso < maxDtSigForPrimaryAssignment_))
        iVertex = vtxIdMinSignif;
    } else {
      // consider only distances to first vertex for association (originally used in PUPPI)
      if ((vtxIdMinSignif >= 0) && (std::abs(track->eta()) > fEtaMinUseDz_) &&
          ((firstVertexDz < maxDzForPrimaryAssignment_ && firstVertexDz / dzE < maxDzSigForPrimaryAssignment_ &&
            track->dzError() < maxDzErrorForPrimaryAssignment_) &&
           (!useTime || std::abs(time - vertices[0].t()) / timeReso < maxDtSigForPrimaryAssignment_)))
        iVertex = 0;
    }
  }

  if (iVertex >= 0)
    return {iVertex, PrimaryVertexAssignment::PrimaryDz};

  // if track not assigned yet, it could be a b-decay secondary , use jet axis dist criterion
  // first find the closest jet within maxJetDeltaR_
  int jetIdx = -1;
  double minDeltaR2 = maxJetDeltaR_ * maxJetDeltaR_;
  for (edm::View<reco::Candidate>::const_iterator ij = jets.begin(); ij != jets.end(); ++ij) {
    if (ij->pt() < minJetPt_)
      continue;  // skip jets below the jet Pt threshold

    double deltaR2 = reco::deltaR2(*ij, *track);
    if (deltaR2 < minDeltaR2 and track->dzError() < maxDzErrorForPrimaryAssignment_) {
      minDeltaR2 = deltaR2;
      jetIdx = std::distance(jets.begin(), ij);
    }
  }
  // if jet found
  if (jetIdx != -1) {
    reco::TransientTrack transientTrack = builder.build(*track);
    GlobalVector direction(jets.at(jetIdx).px(), jets.at(jetIdx).py(), jets.at(jetIdx).pz());
    // find the vertex with the smallest distanceToJetAxis that is still within maxDistaneToJetAxis_
    int vtxIdx = -1;
    double minDistanceToJetAxis = maxDistanceToJetAxis_;
    for (IV iv = vertices.begin(); iv != vertices.end(); ++iv) {
      // only check for vertices that are close enough in Z and for tracks that have not too high dXY
      if (std::abs(track->dz(iv->position())) > maxDzForJetAxisAssigment_ ||
          std::abs(track->dxy(iv->position())) > maxDxyForJetAxisAssigment_)
        continue;

      double distanceToJetAxis = IPTools::jetTrackDistance(transientTrack, direction, *iv).second.value();
      if (distanceToJetAxis < minDistanceToJetAxis) {
        minDistanceToJetAxis = distanceToJetAxis;
        vtxIdx = std::distance(vertices.begin(), iv);
      }
    }
    if (vtxIdx >= 0) {
      iVertex = vtxIdx;
    }
  }
  if (iVertex >= 0)
    return {iVertex, PrimaryVertexAssignment::BTrack};

  // if the track is not compatible with other PVs but is compatible with the BeamSpot, we may simply have not reco'ed the PV!
  //  we still point it to the closest in Z, but flag it as possible orphan-primary
  if (!vertices.empty() && std::abs(track->dxy(vertices[0].position())) < maxDxyForNotReconstructedPrimary_ &&
      std::abs(track->dxy(vertices[0].position()) / track->dxyError()) < maxDxySigForNotReconstructedPrimary_)
    return {vtxIdMinSignif, PrimaryVertexAssignment::NotReconstructedPrimary};

  //FIXME: here we could better handle V0s and NucInt

  // all other tracks could be non-B secondaries and we just attach them with closest Z
  if (vtxIdMinSignif >= 0)
    return {vtxIdMinSignif, PrimaryVertexAssignment::OtherDz};
  //If for some reason even the dz failed (when?) we consider the track not assigned
  return {-1, PrimaryVertexAssignment::Unassigned};
}

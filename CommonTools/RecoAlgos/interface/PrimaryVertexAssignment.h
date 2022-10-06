#ifndef CommonTools_PFCandProducer_PrimaryVertexAssignment_
#define CommonTools_PFCandProducer_PrimaryVertexAssignment_

#include "DataFormats/RecoCandidate/interface/RecoChargedRefCandidate.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

class PrimaryVertexAssignment {
public:
  enum Quality {
    UsedInFit = 7,
    PrimaryDz = 6,
    PrimaryV0 = 5,
    BTrack = 4,
    Unused = 3,
    OtherDz = 2,
    NotReconstructedPrimary = 1,
    Unassigned = 0
  };

  PrimaryVertexAssignment(const edm::ParameterSet& iConfig)
      : maxDzSigForPrimaryAssignment_(iConfig.getParameter<double>("maxDzSigForPrimaryAssignment")),
        maxDzForPrimaryAssignment_(iConfig.getParameter<double>("maxDzForPrimaryAssignment")),
        maxDzErrorForPrimaryAssignment_(iConfig.getParameter<double>("maxDzErrorForPrimaryAssignment")),
        maxDtSigForPrimaryAssignment_(iConfig.getParameter<double>("maxDtSigForPrimaryAssignment")),
        maxJetDeltaR_(iConfig.getParameter<double>("maxJetDeltaR")),
        minJetPt_(iConfig.getParameter<double>("minJetPt")),
        maxDistanceToJetAxis_(iConfig.getParameter<double>("maxDistanceToJetAxis")),
        maxDzForJetAxisAssigment_(iConfig.getParameter<double>("maxDzForJetAxisAssigment")),
        maxDxyForJetAxisAssigment_(iConfig.getParameter<double>("maxDxyForJetAxisAssigment")),
        maxDxySigForNotReconstructedPrimary_(iConfig.getParameter<double>("maxDxySigForNotReconstructedPrimary")),
        maxDxyForNotReconstructedPrimary_(iConfig.getParameter<double>("maxDxyForNotReconstructedPrimary")),
        useTiming_(iConfig.getParameter<bool>("useTiming")),
        useVertexFit_(iConfig.getParameter<bool>("useVertexFit")),
        preferHighRanked_(iConfig.getParameter<bool>("preferHighRanked")),
        fNumOfPUVtxsForCharged_(iConfig.getParameter<unsigned int>("NumOfPUVtxsForCharged")),
        fDzCutForChargedFromPUVtxs_(iConfig.getParameter<double>("DzCutForChargedFromPUVtxs")),
        fPtMaxCharged_(iConfig.getParameter<double>("PtMaxCharged")),
        fEtaMinUseDz_(iConfig.getParameter<double>("EtaMinUseDz")),
        fOnlyUseFirstDz_(iConfig.getParameter<bool>("OnlyUseFirstDz")) {}

  ~PrimaryVertexAssignment() {}

  std::pair<int, PrimaryVertexAssignment::Quality> chargedHadronVertex(
      const reco::VertexCollection& vertices,
      const reco::TrackRef& trackRef,
      const reco::Track* track,
      float trackTime,
      float trackTimeResolution,  // <0 if timing not available for this object
      const edm::View<reco::Candidate>& jets,
      const TransientTrackBuilder& builder) const;

  std::pair<int, PrimaryVertexAssignment::Quality> chargedHadronVertex(
      const reco::VertexCollection& vertices,
      int iVertex,
      const reco::Track* track,
      float trackTime,
      float trackTimeResolution,  // <0 if timing not available for this object
      const edm::View<reco::Candidate>& jets,
      const TransientTrackBuilder& builder) const;

  std::pair<int, PrimaryVertexAssignment::Quality> chargedHadronVertex(
      const reco::VertexCollection& vertices,
      const reco::TrackRef& trackRef,
      float trackTime,
      float trackTimeResolution,  // <0 if timing not available for this object
      const edm::View<reco::Candidate>& jets,
      const TransientTrackBuilder& builder) const {
    return chargedHadronVertex(vertices, trackRef, &(*trackRef), trackTime, trackTimeResolution, jets, builder);
  }

  std::pair<int, PrimaryVertexAssignment::Quality> chargedHadronVertex(const reco::VertexCollection& vertices,
                                                                       const reco::PFCandidate& pfcand,
                                                                       const edm::View<reco::Candidate>& jets,
                                                                       const TransientTrackBuilder& builder) const {
    float time = 0, timeResolution = -1;
    if (useTiming_ && pfcand.isTimeValid()) {
      time = pfcand.time();
      timeResolution = pfcand.timeError();
    }
    if (pfcand.gsfTrackRef().isNull()) {
      if (pfcand.trackRef().isNull())
        return {-1, PrimaryVertexAssignment::Unassigned};
      else
        return chargedHadronVertex(vertices, pfcand.trackRef(), time, timeResolution, jets, builder);
    }
    return chargedHadronVertex(
        vertices, reco::TrackRef(), &(*pfcand.gsfTrackRef()), time, timeResolution, jets, builder);
  }

  std::pair<int, PrimaryVertexAssignment::Quality> chargedHadronVertex(const reco::VertexCollection& vertices,
                                                                       const pat::PackedCandidate& pfcand,
                                                                       const edm::View<reco::Candidate>& jets,
                                                                       const TransientTrackBuilder& builder) const {
    float time = 0, timeResolution = -1;
    if (useTiming_ && pfcand.timeError() > 0) {
      time = pfcand.time();
      timeResolution = pfcand.timeError();
    }
    if (!pfcand.hasTrackDetails())
      return {-1, PrimaryVertexAssignment::Unassigned};
    else
      return chargedHadronVertex(
          vertices,
          (useVertexFit_ && (pfcand.pvAssociationQuality() >= pat::PackedCandidate::UsedInFitLoose))
              ? pfcand.vertexRef().key()
              : -1,
          &pfcand.pseudoTrack(),
          time,
          timeResolution,
          jets,
          builder);
  }

  std::pair<int, PrimaryVertexAssignment::Quality> chargedHadronVertex(const reco::VertexCollection& vertices,
                                                                       const reco::RecoChargedRefCandidate& chcand,
                                                                       const edm::ValueMap<float>* trackTimeTag,
                                                                       const edm::ValueMap<float>* trackTimeResoTag,
                                                                       const edm::View<reco::Candidate>& jets,
                                                                       const TransientTrackBuilder& builder) const {
    float time = 0, timeResolution = -1;
    if (useTiming_) {
      time = (*trackTimeTag)[chcand.track()];
      timeResolution = (*trackTimeResoTag)[chcand.track()];
    }
    if (chcand.track().isNull())
      return {-1, PrimaryVertexAssignment::Unassigned};
    return chargedHadronVertex(vertices, chcand.track(), time, timeResolution, jets, builder);
  }

private:
  double maxDzSigForPrimaryAssignment_;
  double maxDzForPrimaryAssignment_;
  double maxDzErrorForPrimaryAssignment_;
  double maxDtSigForPrimaryAssignment_;
  double maxJetDeltaR_;
  double minJetPt_;
  double maxDistanceToJetAxis_;
  double maxDzForJetAxisAssigment_;
  double maxDxyForJetAxisAssigment_;
  double maxDxySigForNotReconstructedPrimary_;
  double maxDxyForNotReconstructedPrimary_;
  bool useTiming_;
  bool useVertexFit_;
  bool preferHighRanked_;
  unsigned int fNumOfPUVtxsForCharged_;
  double fDzCutForChargedFromPUVtxs_;
  double fPtMaxCharged_;
  double fEtaMinUseDz_;
  bool fOnlyUseFirstDz_;
};

#endif

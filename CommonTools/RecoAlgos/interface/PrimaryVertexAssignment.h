#ifndef CommonTools_PFCandProducer_PrimaryVertexAssignment_
#define CommonTools_PFCandProducer_PrimaryVertexAssignment_

#include "DataFormats/RecoCandidate/interface/RecoChargedRefCandidate.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

class PrimaryVertexAssignment {
 public:
  enum Quality {UsedInFit=7,PrimaryDz=6,PrimaryV0=5,BTrack=4,Unused=3,OtherDz=2,NotReconstructedPrimary=1,Unassigned=0};
 
  PrimaryVertexAssignment(const edm::ParameterSet& iConfig):
   maxDzSigForPrimaryAssignment_(iConfig.getParameter<double>("maxDzSigForPrimaryAssignment")),
   maxDzForPrimaryAssignment_(iConfig.getParameter<double>("maxDzForPrimaryAssignment")),
   maxDzErrorForPrimaryAssignment_(iConfig.getParameter<double>("maxDzErrorForPrimaryAssignment")),
   maxDtSigForPrimaryAssignment_(iConfig.getParameter<double>("maxDtSigForPrimaryAssignment")),   
   maxJetDeltaR_(iConfig.getParameter<double>("maxJetDeltaR")),
   minJetPt_(iConfig.getParameter<double>("minJetPt")),
   maxDistanceToJetAxis_(iConfig.getParameter<double>("maxDistanceToJetAxis")),
   maxDzForJetAxisAssigment_(iConfig.getParameter<double>("maxDzForJetAxisAssigment")),
   maxDxyForJetAxisAssigment_(iConfig.getParameter<double>("maxDxyForJetAxisAssigment")),
   maxDxySigForNotReconstructedPrimary_(iConfig.getParameter<double>("maxDxySigForNotReconstructedPrimary")),
   maxDxyForNotReconstructedPrimary_(iConfig.getParameter<double>("maxDxyForNotReconstructedPrimary"))
  {}

  ~PrimaryVertexAssignment(){}

  std::pair<int,PrimaryVertexAssignment::Quality> chargedHadronVertex(const reco::VertexCollection& vertices, 
               const reco::TrackRef& trackRef,
               const reco::Track * track,
               const edm::ValueMap<float> *trackTimeTag,
               const edm::ValueMap<float> *trackTimeResoTag,
               const edm::View<reco::Candidate> & jets,
              const TransientTrackBuilder & builder) const;

  std::pair<int,PrimaryVertexAssignment::Quality> chargedHadronVertex(const reco::VertexCollection& vertices,
               const reco::TrackRef& trackRef,
               const edm::ValueMap<float> *trackTimeTag,
               const edm::ValueMap<float> *trackTimeResoTag,
               const edm::View<reco::Candidate> & jets,
              const TransientTrackBuilder & builder) const
 {
	return chargedHadronVertex(vertices,trackRef,&(*trackRef),trackTimeTag,trackTimeResoTag,jets,builder);
 }

  std::pair<int,PrimaryVertexAssignment::Quality> chargedHadronVertex( const reco::VertexCollection& vertices,
                                   const reco::PFCandidate& pfcand,
                                   const edm::ValueMap<float> *trackTimeTag,
                                   const edm::ValueMap<float> *trackTimeResoTag,
                                   const edm::View<reco::Candidate>& jets,
                                   const TransientTrackBuilder& builder) const {
	  if(pfcand.gsfTrackRef().isNull())
	  {
		  if(pfcand.trackRef().isNull())
			  return std::pair<int,PrimaryVertexAssignment::Quality>(-1,PrimaryVertexAssignment::Unassigned);
		  else 
			  return chargedHadronVertex(vertices,pfcand.trackRef(),trackTimeTag,trackTimeResoTag,jets,builder);
	  }
	  return chargedHadronVertex(vertices,reco::TrackRef(),&(*pfcand.gsfTrackRef()),trackTimeTag,trackTimeResoTag,jets,builder);
  }
  std::pair<int,PrimaryVertexAssignment::Quality> chargedHadronVertex( const reco::VertexCollection& vertices,
                                   const reco::RecoChargedRefCandidate& chcand,
                                   const edm::ValueMap<float> *trackTimeTag,
                                   const edm::ValueMap<float> *trackTimeResoTag,
                                   const edm::View<reco::Candidate>& jets,
                                   const TransientTrackBuilder& builder) const {
      if(chcand.track().isNull())
         return std::pair<int,PrimaryVertexAssignment::Quality>(-1,PrimaryVertexAssignment::Unassigned);
      return chargedHadronVertex(vertices,chcand.track(),trackTimeTag,trackTimeResoTag,jets,builder);
  }


 private  :
    double    maxDzSigForPrimaryAssignment_;
    double    maxDzForPrimaryAssignment_;
    double    maxDzErrorForPrimaryAssignment_;
    double    maxDtSigForPrimaryAssignment_;
    double    maxJetDeltaR_;
    double    minJetPt_;
    double    maxDistanceToJetAxis_;
    double    maxDzForJetAxisAssigment_;
    double    maxDxyForJetAxisAssigment_;
    double    maxDxySigForNotReconstructedPrimary_;
    double    maxDxyForNotReconstructedPrimary_;
};

#endif

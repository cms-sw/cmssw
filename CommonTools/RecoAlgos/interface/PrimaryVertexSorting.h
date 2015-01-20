#ifndef CommonTools_PFCandProducer_PrimaryVertexSorting_
#define CommonTools_PFCandProducer_PrimaryVertexSorting_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "CommonTools/ParticleFlow/interface/PFPrimaryVertexAssignment.h"



class PrimaryVertexSorting {
 public:
  enum Quality {UsedInFit=0,PrimaryDz,BTrack,OtherDz,NotReconstructedPrimary,Unassigned=99};
 
  PrimaryVertexSorting(const edm::ParameterSet& iConfig)
   //minJetPt_(iConfig.getParameter<double>("minJetPt")),
  {}

  ~PrimaryVertexSorting(){}
  float score(const reco::Vertex & pv, const std::vector<const reco::Candidate *> & candidates , bool useMet) const ;




 private  :
};

#endif

#ifndef RecoParticleFlow_PFTracking_PFDisplacedVertexCandidateProducer_h_
#define RecoParticleFlow_PFTracking_PFDisplacedVertexCandidateProducer_h_

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFTracking/interface/PFDisplacedVertexCandidateFinder.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

/**\class PFDisplacedVertexCandidateProducer 
\brief Producer for DisplacedVertices 

This producer makes use of DisplacedVertexCandidateFinder. This Finder
loop recursively over reco::Tracks to find those which are linked 
together by the criterion which is by default the minimal approach distance. 

\author Maxime Gouzevitch
\date   November 2009
*/

class PFDisplacedVertexCandidateProducer : public edm::stream::EDProducer<> {
 public:

  explicit PFDisplacedVertexCandidateProducer(const edm::ParameterSet&);

  ~PFDisplacedVertexCandidateProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  /// Reco Tracks used to spot the nuclear interactions
  edm::EDGetTokenT<reco::TrackCollection>   inputTagTracks_;
 
  /// Input tag for main vertex to cut of dxy of secondary tracks
  edm::EDGetTokenT<reco::VertexCollection>  inputTagMainVertex_; 
  edm::EDGetTokenT<reco::BeamSpot>    inputTagBeamSpot_;
  
  /// verbose ?
  bool   verbose_;

  /// Displaced Vertex Candidates finder
  PFDisplacedVertexCandidateFinder            pfDisplacedVertexCandidateFinder_;

};

#endif

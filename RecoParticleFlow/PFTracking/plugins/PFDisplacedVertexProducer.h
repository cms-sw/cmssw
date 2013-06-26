#ifndef RecoParticleFlow_PFTracking_PFDisplacedVertexProducer_h_
#define RecoParticleFlow_PFTracking_PFDisplacedVertexProducer_h_

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFTracking/interface/PFDisplacedVertexFinder.h"

/**\class PFDisplacedVertexProducer 
\brief Producer for DisplacedVertices 

This producer makes use of DisplacedVertexFinder. This Finder fit vertex candidates
out of the DisplacedVertexCandidates which contain all tracks linked 
together by the criterion which is by default the minimal approach distance. 

\author Maxime Gouzevitch
\date   November 2009
*/

class PFDisplacedVertexProducer : public edm::EDProducer {
 public:

  explicit PFDisplacedVertexProducer(const edm::ParameterSet&);

  ~PFDisplacedVertexProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  /// Collection of DisplacedVertex Candidates used as input for
  /// the Displaced VertexFinder.
  edm::InputTag   inputTagVertexCandidates_;

  /// Input tag for main vertex to cut of dxy of secondary tracks
  edm::InputTag   inputTagMainVertex_; 
  edm::InputTag   inputTagBeamSpot_;
  
  /// verbose ?
  bool   verbose_;

  /// Displaced Vertices finder
  PFDisplacedVertexFinder            pfDisplacedVertexFinder_;

};

#endif

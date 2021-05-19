#ifndef CommonTools_ParticleFlow_PFnoPileUp_
#define CommonTools_ParticleFlow_PFnoPileUp_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/Association.h"

/**\class PFnoPileUp
\brief Identifies pile-up candidates from a collection of Candidates, and
produces the corresponding collection of NoPileUpCandidates.

\author Andreas Hinzmann
\date   May 2021

*/

class PFnoPileUp : public edm::stream::EDProducer<> {
public:
  typedef edm::View<reco::Candidate> CandidateView;
  typedef edm::Association<reco::VertexCollection> CandToVertex;

  explicit PFnoPileUp(const edm::ParameterSet&);

  ~PFnoPileUp() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<CandidateView> tokenCandidatesView_;
  edm::EDGetTokenT<reco::VertexCollection> tokenVertices_;
  edm::EDGetTokenT<CandToVertex> tokenVertexAssociation_;
  edm::EDGetTokenT<edm::ValueMap<int>> tokenVertexAssociationQuality_;
  int vertexAssociationQuality_;
};

#endif

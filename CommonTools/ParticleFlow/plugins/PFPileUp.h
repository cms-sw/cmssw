#ifndef PhysicsTools_PFCandProducer_PFPileUp_
#define PhysicsTools_PFCandProducer_PFPileUp_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CommonTools/ParticleFlow/interface/PFPileUpAlgo.h"

/**\class PFPileUp
\brief Identifies pile-up candidates from a collection of PFCandidates, and
produces the corresponding collection of PileUpCandidates.

\author Colin Bernet
\date   february 2008
\updated Florian Beaudette 30/03/2012

*/




class PFPileUp : public edm::EDProducer {
 public:

  typedef std::vector< edm::FwdPtr<reco::PFCandidate> >  PFCollection;
  typedef edm::View<reco::PFCandidate>                   PFView;
  typedef std::vector<reco::PFCandidate>                 PFCollectionByValue;

  explicit PFPileUp(const edm::ParameterSet&);

  ~PFPileUp();

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  PFPileUpAlgo    pileUpAlgo_;

  /// PFCandidates to be analyzed
  edm::EDGetTokenT<PFCollection>   tokenPFCandidates_;

  /// vertices
  edm::EDGetTokenT<reco::VertexCollection>   tokenVertices_;

  /// enable PFPileUp selection
  bool   enable_;

  /// verbose ?
  bool   verbose_;

  /// use the closest z vertex if a track is not in a vertex
  bool   checkClosestZVertex_;

};

#endif

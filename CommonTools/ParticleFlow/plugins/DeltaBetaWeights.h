
// Weight for neutral particles based on distance with charged
// 
// Original Author:  Michail Bachtis,40 1-B08,+41227678176,
//         Created:  Mon Dec  9 13:18:05 CET 2013
//
// edited by Pavel Jez
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

//
// class declaration
//

class DeltaBetaWeights : public edm::EDProducer {
 public:
  explicit DeltaBetaWeights(const edm::ParameterSet&);
  ~DeltaBetaWeights();

 private:

  virtual void produce(edm::Event&, const edm::EventSetup&);
  // ----------member data ---------------------------
  edm::InputTag src_;
  edm::InputTag pfCharged_;
  edm::InputTag pfPU_;
  
  edm::EDGetTokenT<edm::View<reco::Candidate> > pfCharged_token;
  edm::EDGetTokenT<edm::View<reco::Candidate> > pfPU_token;
  edm::EDGetTokenT<edm::View<reco::Candidate> > src_token;



};

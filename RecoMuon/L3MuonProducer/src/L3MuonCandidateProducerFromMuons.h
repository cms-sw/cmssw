#ifndef RecoMuon_L3MuonProducer_L3MuonCandidateProducerFromMuons_H
#define RecoMuon_L3MuonProducer_L3MuonCandidateProducerFromMuons_H

/**  \class L3MuonCandidateProducerFromMuons
 * 
 *   Intermediate step in the L3 muons selection.
 *   This class takes the tracker muons (which are reco::Muons) 
 *   and creates the correspondent reco::RecoChargedCandidate.
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class L3MuonCandidateProducerFromMuons : public edm::EDProducer {

 public:

  /// constructor with config
  L3MuonCandidateProducerFromMuons(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L3MuonCandidateProducerFromMuons(); 
  
  /// produce candidates
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  
  // L3/GLB Collection Label
  edm::InputTag m_L3CollectionLabel; 
};

#endif

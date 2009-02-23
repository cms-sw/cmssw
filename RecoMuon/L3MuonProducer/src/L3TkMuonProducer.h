#ifndef RecoMuon_L3MuonProducer_L3TkMuonProducer_H
#define RecoMuon_L3MuonProducer_L3TkMuonProducer_H

/**  \class L3TkMuonProducer
 * 
 *    This module creates a skimed list of reco::Track (pointing to the original TrackExtra and TrackingRecHitOwnedVector
 *    One highest pT track per L1/L2 is selected, requiring some quality.
 *
 *   \author  J-R Vlimant.
 */


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class L3TkMuonProducer : public edm::EDProducer {

 public:

  /// constructor with config
  L3TkMuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L3TkMuonProducer(); 
  
  /// produce candidates
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  
  // L3/GLB Collection Label
  edm::InputTag theL3CollectionLabel; 

};

#endif

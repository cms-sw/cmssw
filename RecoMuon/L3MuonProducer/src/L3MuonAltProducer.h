#ifndef RecoMuon_L3MuonIsolationProducer_L3MuonAltProducer_H
#define RecoMuon_L3MuonIsolationProducer_L3MuonAltProducer_H

/**  \class L3MuonAltProducer
 * 
 *   \author  J.Alcaraz
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class L3MuonAltProducer : public edm::EDProducer {

 public:

  /// constructor with config
  L3MuonAltProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L3MuonAltProducer(); 
  
  /// produce candidates
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
  /// match candidates
  reco::TrackRef muonMatch(const reco::TrackRef& mu, const edm::Handle<reco::TrackCollection>& trks) const;
  
 private:
  
  // Regional Track Collection Label
  edm::InputTag theTrackCollectionLabel; 

  // Muon Collection Label
  edm::InputTag theMuonCollectionLabel; 

  // Maximum Chi2 per dof allowed in muon-track matching 
  double theMaxChi2PerDof;

};
  
#endif

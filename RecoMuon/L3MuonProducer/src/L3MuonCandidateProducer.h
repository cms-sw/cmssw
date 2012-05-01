#ifndef RecoMuon_L3MuonProducer_L3MuonCandidateProducer_H
#define RecoMuon_L3MuonProducer_L3MuonCandidateProducer_H

/**  \class L3MuonCandidateProducer
 * 
 *   Intermediate step in the L3 muons selection.
 *   This class takes the L3 muons (which are reco::Tracks) 
 *   and creates the correspondent reco::RecoChargedCandidate.
 *
 *   Riccardo's comment:
 *   The separation between the L3MuonProducer and this class allows
 *   the interchangeability of the L3MuonProducer and the StandAloneMuonProducer
 *   This class is supposed to be removed once the
 *   L3/STA comparison will be done, then the RecoChargedCandidate
 *   production will be done into the L3MuonProducer class.
 *
 *
 *   \author  J.Alcaraz
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class L3MuonCandidateProducer : public edm::EDProducer {

 public:
  enum MuonTrackType {InnerTrack, OuterTrack, CombinedTrack};

  /// constructor with config
  L3MuonCandidateProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L3MuonCandidateProducer(); 
  
  /// produce candidates
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  // L3/GLB Collection Label
  edm::InputTag theL3CollectionLabel; 
  edm::InputTag theL3LinksLabel; 
  enum MuonTrackType theType;
  bool theUseLinks;
};

#endif

#ifndef RecoMuon_L2MuonProducer_L2MuonCandidateProducer_H
#define RecoMuon_L2MuonProducer_L2MuonCandidateProducer_H

/**  \class L2MuonCandidateProducer
 * 
 *   Intermediate step in the L2 muons selection.
 *   This class takes the L2 muons (which are reco::Tracks) 
 *   and creates the correspondent reco::RecoChargedCandidate.
 *
 *   Riccardo's comment:
 *   The separation between the L2MuonProducer and this class allows
 *   the interchangeability of the L2MuonProducer and the StandAloneMuonProducer
 *   This class is supposed to be removed once the
 *   L2/STA comparison will be done, then the RecoChargedCandidate
 *   production will be done into the L2MuonProducer class.
 *
 *
 *   \author  J.Alcaraz
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class L2MuonCandidateProducer : public edm::EDProducer {

 public:

  /// constructor with config
  L2MuonCandidateProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L2MuonCandidateProducer(); 
  
  /// produce candidates
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  
  // StandAlone Collection Label
  edm::InputTag theSACollectionLabel; 
};

#endif

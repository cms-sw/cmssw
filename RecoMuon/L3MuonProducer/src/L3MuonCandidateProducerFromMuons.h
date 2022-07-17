#ifndef RecoMuon_L3MuonProducer_L3MuonCandidateProducerFromMuons_H
#define RecoMuon_L3MuonProducer_L3MuonCandidateProducerFromMuons_H

/**  \class L3MuonCandidateProducerFromMuons
 * 
 *   Intermediate step in the L3 muons selection.
 *   This class takes the tracker muons (which are reco::Muons) 
 *   and creates the correspondent reco::RecoChargedCandidate.
 *
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class L3MuonCandidateProducerFromMuons : public edm::global::EDProducer<> {
public:
  /// constructor with config
  L3MuonCandidateProducerFromMuons(const edm::ParameterSet&);

  /// destructor
  ~L3MuonCandidateProducerFromMuons() override;

  /// descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// produce candidates
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // L3/GLB Collection Label
  edm::InputTag const m_L3CollectionLabel;
  edm::EDGetTokenT<reco::MuonCollection> const m_muonToken;
  bool const m_displacedReco;
};

#endif

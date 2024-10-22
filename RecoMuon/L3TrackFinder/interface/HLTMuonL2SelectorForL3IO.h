#ifndef RecoMuon_HLTMuonL2SelectorForL3IO_HLTMuonL2SelectorForL3IO_H
#define RecoMuon_HLTMuonL2SelectorForL3IO_HLTMuonL2SelectorForL3IO_H

/**  \class HLTMuonL2SelectorForL3IO
 * 
 *   L2 muon selector for L3 IO:
 *   finds L2 muons not previous converted into (good) L3 muons 
 *
 *   \author  Benjamin Radburn-Smith, Santiago Folgueras - Purdue University
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class HLTMuonL2SelectorForL3IO : public edm::stream::EDProducer<> {
public:
  /// constructor with config
  HLTMuonL2SelectorForL3IO(const edm::ParameterSet&);

  /// destructor
  ~HLTMuonL2SelectorForL3IO() override;

  /// default values
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// select muons
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<reco::TrackCollection> l2Src_;
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> l3OISrc_;
  const edm::EDGetTokenT<reco::MuonTrackLinksCollection> l3linkToken_;
  const bool applyL3Filters_;
  const double max_NormalizedChi2_, max_PtDifference_;
  const int min_Nhits_, min_NmuonHits_;
};

#endif

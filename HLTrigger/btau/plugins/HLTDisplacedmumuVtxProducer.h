#ifndef HLTDisplacedmumuVtxProducer_h
#define HLTDisplacedmumuVtxProducer_h

/** \class HLTDisplacedmumuVtxProducer_h
 *
 *  
 *  produces kalman vertices from di-muons
 *  takes muon candidates as input
 *  configurable cuts on pt, eta, pair pt, inv. mass
 *
 *  \author Alexander.Schmidt@cern.ch
 *  \date   3. Feb. 2011
 *
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include <vector>

namespace edm {
  class ConfigurationDescriptions;
}

class HLTDisplacedmumuVtxProducer : public edm::global::EDProducer<> {
public:
  explicit HLTDisplacedmumuVtxProducer(const edm::ParameterSet&);
  ~HLTDisplacedmumuVtxProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  bool checkPreviousCand(const reco::TrackRef& trackref, const std::vector<reco::RecoChargedCandidateRef>& ref2) const;

  const edm::InputTag srcTag_;
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> srcToken_;
  const edm::InputTag previousCandTag_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_;
  const bool matchToPrevious_;
  const double maxEta_;
  const double minPt_;
  const double minPtPair_;
  const double minInvMass_;
  const double maxInvMass_;
  const int chargeOpt_;
};

#endif

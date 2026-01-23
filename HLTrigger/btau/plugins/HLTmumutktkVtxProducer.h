#ifndef HLTrigger_btau_HLTmumutktkVtxProducer_h
#define HLTrigger_btau_HLTmumutktkVtxProducer_h
//
// Package:    HLTrigger/btau
// Class:      HLTmumutktkVtxProducer
//
/**\class HLTmumutktkVtxProducer
*/

#include <vector>
#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateOnlyFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

namespace edm {
  class ConfigurationDescriptions;
}

// ----------------------------------------------------------------------

class FreeTrajectoryState;
class MagneticField;

class HLTmumutktkVtxProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTmumutktkVtxProducer(const edm::ParameterSet&);
  ~HLTmumutktkVtxProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  bool overlap(const reco::TrackRef& trackref1, const reco::TrackRef& trackref2);
  static FreeTrajectoryState initialFreeState(const reco::Track&, const MagneticField*);
  bool checkPreviousCand(const reco::TrackRef& trackref, const std::vector<reco::RecoChargedCandidateRef>& ref2) const;

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackRecordToken_;

  const edm::InputTag muCandTag_;
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> muCandToken_;
  const edm::InputTag trkCandTag_;
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> trkCandToken_;
  const edm::InputTag previousCandTag_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_;

  const std::string mfName_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldRecordToken_;

  const double thirdTrackMass_;
  const double fourthTrackMass_;
  const double maxEta_;
  const double minPt_;
  const double minInvMass_;
  const double maxInvMass_;
  const double minTrkTrkMass_;
  const double maxTrkTrkMass_;
  const double minD0Significance_;
  const bool oppositeSign_;
  const double overlapDR2_;
  const edm::InputTag beamSpotTag_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
};

#endif

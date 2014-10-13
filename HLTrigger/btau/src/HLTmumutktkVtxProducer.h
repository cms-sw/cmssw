#ifndef HLTmumutktkVtxProducer_h
#define HLTmumutktkVtxProducer_h
//
// Package:    HLTstaging
// Class:      HLTmumutktkVtxProducer
//
/**\class HLTmumutktkVtxProducer
*/

// system include files
#include <memory>

#include "FWCore/Framework/interface/EDProducer.h"
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

// ----------------------------------------------------------------------

namespace reco {
  class Candidate;
  class Track;
}

class FreeTrajectoryState;
class MagneticField;
    
class HLTmumutktkVtxProducer : public edm::EDProducer {
 public:
  explicit HLTmumutktkVtxProducer(const edm::ParameterSet&);
  ~HLTmumutktkVtxProducer();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  bool overlap(const reco::TrackRef& trackref1, const reco::TrackRef& trackref2);
  static FreeTrajectoryState initialFreeState( const reco::Track&,const MagneticField*);
  bool checkPreviousCand(const reco::TrackRef& trackref, std::vector<reco::RecoChargedCandidateRef>& ref2);

  edm::InputTag                                          muCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> muCandToken_;
  edm::InputTag                                          trkCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> trkCandToken_;
  edm::InputTag                                          previousCandTag_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_;

  const std::string mfName_;
  const double thirdTrackMass_;
  const double fourthTrackMass_;
  const double maxEta_;
  const double minPt_;
  const double minInvMass_;
  const double maxInvMass_;
  const double minTrkTrkMass_;
  const double maxTrkTrkMass_;
  const double minD0Significance_;
  bool         oppositeSign_;
  const double overlapDR_;
  edm::InputTag                    beamSpotTag_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

};
#endif

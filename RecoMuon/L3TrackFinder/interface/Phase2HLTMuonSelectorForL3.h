#ifndef RecoMuon_L3TrackFinder_phase2HLTMuonSelectorForL3_H
#define RecoMuon_L3TrackFinder_phase2HLTMuonSelectorForL3_H

/**  \class Phase2HLTMuonSelectorForL3
 * 
 *   Phase-2 L3 selector for Muons
 *   This module allows to choose whether to perform 
 *   Inside-Out or Outside-In reconstruction first for L3 Muons, 
 *   performing the second pass only on candidates that were not
 *   reconstructed or whose quality was not good enough. Required 
 *   quality criteria are configurable, the default parameters 
 *   match the requests of HLT Muon ID. 
 *   When Inside-Out reconstruction is performed first, the resulting
 *   L3 Tracks are filtered and geometrically matched with L2
 *   Standalone Muons. If either the match is unsuccessful, or 
 *   the L3 track is not of good-enough quality, the associated 
 *   Standalone Muon will be re-used to seed the Outside-In step.
 *   The Outside-In first approach follows a similar logic by 
 *   matching the L3 tracks directly with L1 Tracker Muons. 
 *   Then, when either the match fails or the track is not of 
 *   good-enough quality, the L1 Tracker Muon is re-used to seed
 *   the Inside-Out reconstruction.
 *
 *   \author Luca Ferragina (INFN BO), 2024
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // End namespace edm

class Phase2HLTMuonSelectorForL3 : public edm::stream::EDProducer<> {
public:
  // Constructor
  Phase2HLTMuonSelectorForL3(const edm::ParameterSet&);

  // Destructor
  ~Phase2HLTMuonSelectorForL3() override = default;

  // Default values
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // Select objects to be reused
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<l1t::TrackerMuonCollection> l1TkMuCollToken_;
  const edm::EDGetTokenT<reco::TrackCollection> l2MuCollectionToken_;
  const edm::EDGetTokenT<reco::TrackCollection> l3TrackCollectionToken_;

  const bool IOFirst_;
  const double matchingDr_;
  const bool applyL3Filters_;
  const double maxNormalizedChi2_, maxPtDifference_;
  const int minNhits_, minNhitsMuons_, minNhitsPixel_, minNhitsTracker_;

  // Check L3 inner track quality parameters
  const bool rejectL3Track(l1t::TrackerMuonRef l1TkMuRef, reco::TrackRef l3TrackRef) const;
};

#endif

// -*- C++ -*-
//
// Package:    MuonAlignmentPreFilter
// Class:      MuonAlignmentPreFilter
//
/**\class MuonAlignmentPreFilter

 Description: pre-select events that are worth considering in muon alignment 

 $Id:$
*/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

class MuonAlignmentPreFilter : public edm::stream::EDFilter<> {
public:
  explicit MuonAlignmentPreFilter(const edm::ParameterSet&);
  ~MuonAlignmentPreFilter() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const edm::InputTag m_tracksTag;
  const double m_minTrackPt;
  const double m_minTrackP;
  const bool m_allowTIDTEC;
  const int m_minTrackerHits;
  const int m_minDTHits;
  const int m_minCSCHits;
  const double m_minTrackEta;
  const double m_maxTrackEta;
  const edm::EDGetTokenT<reco::TrackCollection> m_trackToken;
};

MuonAlignmentPreFilter::MuonAlignmentPreFilter(const edm::ParameterSet& cfg)
    : m_tracksTag(cfg.getParameter<edm::InputTag>("tracksTag")),
      m_minTrackPt(cfg.getParameter<double>("minTrackPt")),
      m_minTrackP(cfg.getParameter<double>("minTrackP")),
      m_allowTIDTEC(cfg.getParameter<bool>("allowTIDTEC")),
      m_minTrackerHits(cfg.getParameter<int>("minTrackerHits")),
      m_minDTHits(cfg.getParameter<int>("minDTHits")),
      m_minCSCHits(cfg.getParameter<int>("minCSCHits")),
      m_minTrackEta(cfg.getParameter<double>("minTrackEta")),
      m_maxTrackEta(cfg.getParameter<double>("maxTrackEta")),
      m_trackToken(consumes<reco::TrackCollection>(m_tracksTag)) {}

void MuonAlignmentPreFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksTag", edm::InputTag("ALCARECOMuAlCalIsolatedMu:GlobalMuon"));
  desc.add<double>("minTrackPt", 20.);
  desc.add<double>("minTrackP", 0.);
  desc.add<int>("minTrackerHits", 10);
  desc.add<int>("minDTHits", 6);
  desc.add<int>("minCSCHits", 4);
  desc.add<bool>("allowTIDTEC", true);
  desc.add<double>("minTrackEta", -2.4);
  desc.add<double>("maxTrackEta", 2.4);
  descriptions.add("MuonAlignmentPreFilter", desc);
}

bool MuonAlignmentPreFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const edm::Handle<reco::TrackCollection>& trackColl = iEvent.getHandle(m_trackToken);

  // check if there's at least one interesting track:

  for (reco::TrackCollection::const_iterator it = trackColl->begin(); it != trackColl->end(); it++) {
    int tracker_numHits = 0;
    bool contains_TIDTEC = false;
    int dt_numHits = 0;
    int csc_numHits = 0;

    const reco::Track* track = &(*it);

    if (track->pt() < m_minTrackPt || track->p() < m_minTrackP)
      continue;
    if (track->eta() < m_minTrackEta || track->eta() > m_maxTrackEta)
      continue;

    for (auto const& hit : track->recHits()) {
      DetId id = hit->geographicalId();
      if (id.det() == DetId::Tracker) {
        tracker_numHits++;
        if (id.subdetId() == StripSubdetector::TID || id.subdetId() == StripSubdetector::TEC)
          contains_TIDTEC = true;
      }

      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT)
        dt_numHits++;
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC)
        csc_numHits++;
    }

    if ((m_allowTIDTEC || !contains_TIDTEC) && m_minTrackerHits <= tracker_numHits &&
        (m_minDTHits <= dt_numHits || m_minCSCHits <= csc_numHits))
      return true;
  }
  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonAlignmentPreFilter);

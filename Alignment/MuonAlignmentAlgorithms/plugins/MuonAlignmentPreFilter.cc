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
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

class MuonAlignmentPreFilter : public edm::EDFilter {
public:
  explicit MuonAlignmentPreFilter(const edm::ParameterSet&);
  ~MuonAlignmentPreFilter() override {}

private:
  void beginJob() override {}
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  // ----------member data ---------------------------

  edm::InputTag m_tracksTag;
  double m_minTrackPt;
  double m_minTrackP;
  bool m_allowTIDTEC;
  int m_minTrackerHits;
  int m_minDTHits;
  int m_minCSCHits;
  double m_minTrackEta;
  double m_maxTrackEta;
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
      m_maxTrackEta(cfg.getParameter<double>("maxTrackEta")) {}

bool MuonAlignmentPreFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::TrackCollection> trackColl;
  iEvent.getByLabel(m_tracksTag, trackColl);

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

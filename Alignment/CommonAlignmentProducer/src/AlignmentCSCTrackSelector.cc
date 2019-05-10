#include "Alignment/CommonAlignmentProducer/interface/AlignmentCSCTrackSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

// constructor ----------------------------------------------------------------

AlignmentCSCTrackSelector::AlignmentCSCTrackSelector(const edm::ParameterSet& cfg)
    : m_src(cfg.getParameter<edm::InputTag>("src")),
      m_stationA(cfg.getParameter<int>("stationA")),
      m_stationB(cfg.getParameter<int>("stationB")),
      m_minHitsDT(cfg.getParameter<int>("minHitsDT")),
      m_minHitsPerStation(cfg.getParameter<int>("minHitsPerStation")),
      m_maxHitsPerStation(cfg.getParameter<int>("maxHitsPerStation")) {}

// destructor -----------------------------------------------------------------

AlignmentCSCTrackSelector::~AlignmentCSCTrackSelector() {}

// do selection ---------------------------------------------------------------

AlignmentCSCTrackSelector::Tracks AlignmentCSCTrackSelector::select(const Tracks& tracks, const edm::Event& evt) const {
  Tracks result;

  for (auto const& track : tracks) {
    int hitsOnStationA = 0;
    int hitsOnStationB = 0;

    for (auto const& hit : track->recHits()) {
      DetId id = hit->geographicalId();

      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT) {
        if (m_stationA == 0)
          hitsOnStationA++;
        if (m_stationB == 0)
          hitsOnStationB++;
      } else if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
        CSCDetId cscid(id.rawId());
        int station = (cscid.endcap() == 1 ? 1 : -1) * cscid.station();

        if (station == m_stationA)
          hitsOnStationA++;
        if (station == m_stationB)
          hitsOnStationB++;

      }  // end if CSC
    }    // end loop over hits

    bool stationAokay;
    if (m_stationA == 0)
      stationAokay = (m_minHitsDT <= hitsOnStationA);
    else
      stationAokay = (m_minHitsPerStation <= hitsOnStationA && hitsOnStationA <= m_maxHitsPerStation);

    bool stationBokay;
    if (m_stationB == 0)
      stationBokay = (m_minHitsDT <= hitsOnStationB);
    else
      stationBokay = (m_minHitsPerStation <= hitsOnStationB && hitsOnStationB <= m_maxHitsPerStation);

    if (stationAokay && stationBokay) {
      result.push_back(track);
    }
  }  // end loop over tracks

  return result;
}

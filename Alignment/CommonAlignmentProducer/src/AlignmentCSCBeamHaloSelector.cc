#include "Alignment/CommonAlignmentProducer/interface/AlignmentCSCBeamHaloSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

// constructor ----------------------------------------------------------------

AlignmentCSCBeamHaloSelector::AlignmentCSCBeamHaloSelector(const edm::ParameterSet &iConfig)
   : m_minStations(iConfig.getParameter<unsigned int>("minStations"))
   , m_minHitsPerStation(iConfig.getParameter<unsigned int>("minHitsPerStation"))
{
   edm::LogInfo("AlignmentCSCBeamHaloSelector") 
      << "Acceptable tracks must have at least " << m_minHitsPerStation << " hits in " << m_minStations << " different CSC stations." << std::endl;
}

// destructor -----------------------------------------------------------------

AlignmentCSCBeamHaloSelector::~AlignmentCSCBeamHaloSelector() {}

// do selection ---------------------------------------------------------------

AlignmentCSCBeamHaloSelector::Tracks 
AlignmentCSCBeamHaloSelector::select(const Tracks &tracks, const edm::Event &iEvent) const {
   Tracks result;

   for (Tracks::const_iterator track = tracks.begin();  track != tracks.end();  ++track) {
      std::map<int, unsigned int> station_map;

      for (trackingRecHit_iterator hit = (*track)->recHitsBegin();  hit != (*track)->recHitsEnd();  ++hit) {
	 DetId id = (*hit)->geographicalId();
	 if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	    CSCDetId cscid(id.rawId());
	    int station = (cscid.endcap() == 1 ? 1 : -1) * cscid.station();

	    std::map<int, unsigned int>::const_iterator station_iter = station_map.find(station);
	    if (station_iter == station_map.end()) {
	       station_map[station] = 0;
	    }
	    station_map[station]++;
	 } // end if it's a CSC hit
      } // end loop over hits

      unsigned int stations = 0;
      for (std::map<int, unsigned int>::const_iterator station_iter = station_map.begin();  station_iter != station_map.end();  ++station_iter) {
	 if (station_iter->second > m_minHitsPerStation) stations++;
      }
      if (stations >= m_minStations) {
	 result.push_back(*track);
      }
   } // end loop over tracks
  
   return result;
}

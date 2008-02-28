#include "Alignment/CommonAlignmentProducer/interface/AlignmentCSCOverlapSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

// constructor ----------------------------------------------------------------

AlignmentCSCOverlapSelector::AlignmentCSCOverlapSelector(const edm::ParameterSet &iConfig)
   : m_station(iConfig.getParameter<int>("station"))
   , m_minHitsPerChamber(iConfig.getParameter<unsigned int>("minHitsPerChamber"))
{
   if (m_station == 0) {
      edm::LogInfo("AlignmentCSCOverlapSelector") 
	 << "Acceptable tracks must have " << m_minHitsPerChamber << " in two chambers on all stations." << std::endl;
   }
   else {
      edm::LogInfo("AlignmentCSCOverlapSelector") 
	 << "Acceptable tracks must have " << m_minHitsPerChamber << " in two chambers on station " << m_station << "." << std::endl;
   }
}

// destructor -----------------------------------------------------------------

AlignmentCSCOverlapSelector::~AlignmentCSCOverlapSelector() {}

// do selection ---------------------------------------------------------------

AlignmentCSCOverlapSelector::Tracks 
AlignmentCSCOverlapSelector::select(const Tracks &tracks, const edm::Event &iEvent) const {
   Tracks result;

   for (Tracks::const_iterator track = tracks.begin();  track != tracks.end();  ++track) {
      unsigned int even = 0;
      unsigned int odd = 0;

      for (trackingRecHit_iterator hit = (*track)->recHitsBegin();  hit != (*track)->recHitsEnd();  ++hit) {
	 DetId id = (*hit)->geographicalId();
	 if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	    CSCDetId cscid(id.rawId());
	    int station = (cscid.endcap() == 1 ? 1 : -1) * cscid.station();

	    if (m_station == 0  ||  station == m_station) {
	       if (cscid.chamber() % 2 == 0) even++;
	       else odd++;
	    }
	 } // end if it's a CSC hit
      } // end loop over hits

      if ((even >= m_minHitsPerChamber)  &&  (odd >= m_minHitsPerChamber)) {
	 result.push_back(*track);
      }
   } // end loop over tracks
  
   return result;
}

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
    : m_station(iConfig.getParameter<int>("station")),
      m_minHitsPerChamber(iConfig.getParameter<unsigned int>("minHitsPerChamber")) {
  if (m_station == 0) {
    edm::LogInfo("AlignmentCSCOverlapSelector")
        << "Acceptable tracks must have " << m_minHitsPerChamber << " in two chambers on all stations." << std::endl;
  } else {
    edm::LogInfo("AlignmentCSCOverlapSelector") << "Acceptable tracks must have " << m_minHitsPerChamber
                                                << " in two chambers on station " << m_station << "." << std::endl;
  }
}

// destructor -----------------------------------------------------------------

AlignmentCSCOverlapSelector::~AlignmentCSCOverlapSelector() {}

// do selection ---------------------------------------------------------------

AlignmentCSCOverlapSelector::Tracks AlignmentCSCOverlapSelector::select(const Tracks &tracks,
                                                                        const edm::Event &iEvent) const {
  Tracks result;

  for (auto const &track : tracks) {
    unsigned int MEminus4_even = 0;
    unsigned int MEminus4_odd = 0;
    unsigned int MEminus3_even = 0;
    unsigned int MEminus3_odd = 0;
    unsigned int MEminus2_even = 0;
    unsigned int MEminus2_odd = 0;
    unsigned int MEminus1_even = 0;
    unsigned int MEminus1_odd = 0;

    unsigned int MEplus1_even = 0;
    unsigned int MEplus1_odd = 0;
    unsigned int MEplus2_even = 0;
    unsigned int MEplus2_odd = 0;
    unsigned int MEplus3_even = 0;
    unsigned int MEplus3_odd = 0;
    unsigned int MEplus4_even = 0;
    unsigned int MEplus4_odd = 0;

    for (auto const &hit : track->recHits()) {
      DetId id = hit->geographicalId();
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
        CSCDetId cscid(id.rawId());
        int station = (cscid.endcap() == 1 ? 1 : -1) * cscid.station();

        if (station == -4) {
          if (cscid.chamber() % 2 == 0)
            MEminus4_even++;
          else
            MEminus4_odd++;
        } else if (station == -3) {
          if (cscid.chamber() % 2 == 0)
            MEminus3_even++;
          else
            MEminus3_odd++;
        } else if (station == -2) {
          if (cscid.chamber() % 2 == 0)
            MEminus2_even++;
          else
            MEminus2_odd++;
        } else if (station == -1) {
          if (cscid.chamber() % 2 == 0)
            MEminus1_even++;
          else
            MEminus1_odd++;
        }

        else if (station == 1) {
          if (cscid.chamber() % 2 == 0)
            MEplus1_even++;
          else
            MEplus1_odd++;
        } else if (station == 2) {
          if (cscid.chamber() % 2 == 0)
            MEplus2_even++;
          else
            MEplus2_odd++;
        } else if (station == 3) {
          if (cscid.chamber() % 2 == 0)
            MEplus3_even++;
          else
            MEplus3_odd++;
        } else if (station == 4) {
          if (cscid.chamber() % 2 == 0)
            MEplus4_even++;
          else
            MEplus4_odd++;
        }

      }  // end if it's a CSC hit
    }    // end loop over hits

    if ((m_station == 0 || m_station == -4) && (MEminus4_even >= m_minHitsPerChamber) &&
        (MEminus4_odd >= m_minHitsPerChamber))
      result.push_back(track);

    else if ((m_station == 0 || m_station == -3) && (MEminus3_even >= m_minHitsPerChamber) &&
             (MEminus3_odd >= m_minHitsPerChamber))
      result.push_back(track);

    else if ((m_station == 0 || m_station == -2) && (MEminus2_even >= m_minHitsPerChamber) &&
             (MEminus2_odd >= m_minHitsPerChamber))
      result.push_back(track);

    else if ((m_station == 0 || m_station == -1) && (MEminus1_even >= m_minHitsPerChamber) &&
             (MEminus1_odd >= m_minHitsPerChamber))
      result.push_back(track);

    else if ((m_station == 0 || m_station == 1) && (MEplus1_even >= m_minHitsPerChamber) &&
             (MEplus1_odd >= m_minHitsPerChamber))
      result.push_back(track);

    else if ((m_station == 0 || m_station == 2) && (MEplus2_even >= m_minHitsPerChamber) &&
             (MEplus2_odd >= m_minHitsPerChamber))
      result.push_back(track);

    else if ((m_station == 0 || m_station == 3) && (MEplus3_even >= m_minHitsPerChamber) &&
             (MEplus3_odd >= m_minHitsPerChamber))
      result.push_back(track);

    else if ((m_station == 0 || m_station == 4) && (MEplus4_even >= m_minHitsPerChamber) &&
             (MEplus4_odd >= m_minHitsPerChamber))
      result.push_back(track);

  }  // end loop over tracks

  return result;
}

#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsFromTrack_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsFromTrack_H

/** \class MuonResidualsFromTrack
 *  $Date: 2009/02/27 18:58:29 $
 *  $Revision: 1.2 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include <vector>
#include <map>

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonDT13ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonDT2ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonCSCChamberResidual.h"

class MuonResidualsFromTrack {
public:
  MuonResidualsFromTrack(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, const Trajectory *traj, AlignableNavigator *navigator, double maxResidual);
  ~MuonResidualsFromTrack();

  int trackerNumHits() const { return m_tracker_numHits; };
  double trackerChi2() const { return m_tracker_chi2; };
  double trackerRedChi2() const {
    if (m_tracker_numHits > 5) return m_tracker_chi2 / double(m_tracker_numHits - 5);
    else return -1.;
  };

  bool contains_TIDTEC() const { return m_contains_TIDTEC; };

  const std::vector<DetId> chamberIds() const { return m_chamberIds; };

  MuonChamberResidual *chamberResidual(DetId chamberId, int type) {
    if (type == MuonChamberResidual::kDT13) {
      if (m_dt13.find(chamberId) == m_dt13.end()) return NULL;
      return m_dt13[chamberId];
    }
    else if (type == MuonChamberResidual::kDT2) {
      if (m_dt2.find(chamberId) == m_dt2.end()) return NULL;
      return m_dt2[chamberId];
    }
    else if (type == MuonChamberResidual::kCSC) {
      if (m_csc.find(chamberId) == m_csc.end()) return NULL;
      return m_csc[chamberId];
    }
    else return NULL;
  };

private:
  TrajectoryStateCombiner m_tsoscomb;

  int m_tracker_numHits;
  double m_tracker_chi2;
  bool m_contains_TIDTEC;

  std::vector<DetId> m_chamberIds;
  std::map<DetId,MuonChamberResidual*> m_dt13, m_dt2, m_csc;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsFromTrack_H

#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsFromTrack_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsFromTrack_H

/** \class MuonResidualsFromTrack
 *  $Date: 2009/02/02 13:46:01 $
 *  $Revision: 1.1 $
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

  const std::vector<unsigned int> indexes() const { return m_indexes; };

  MuonChamberResidual *chamberResidual(unsigned int chamberId) {
    if (m_chamberResiduals.find(chamberId) == m_chamberResiduals.end()) return NULL;
    return m_chamberResiduals[chamberId];
  };

private:
  TrajectoryStateCombiner m_tsoscomb;

  int m_tracker_numHits;
  double m_tracker_chi2;
  bool m_contains_TIDTEC;

  std::vector<unsigned int> m_indexes;
  std::map<unsigned int,MuonChamberResidual*> m_chamberResiduals;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsFromTrack_H

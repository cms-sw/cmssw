#ifndef Alignment_MuonAlignmentAlgorithms_MuonTrackCSCChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonTrackCSCChamberResidual_H

/** \class MuonTrackCSCChamberResidual
 * 
 * Implementation of tracker muon chamber residuals for CSC
 * 
 * $Id: $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"

class MuonTrackCSCChamberResidual: public MuonChamberResidual
{
public:
  MuonTrackCSCChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                         DetId chamberId, AlignableDetOrUnitPtr chamberAlignable);
  
  // dummy method
  void addResidual(edm::ESHandle<Propagator> prop, const TrajectoryStateOnSurface *tsos, const TrackingRecHit *hit,double, double) override = 0;

  // for CSC, the residual is chamber local x, projected by the strip measurement direction
  // for CSC, the resslope is dresx/dz, or tan(phi_y)
  void setSegmentResidual(const reco::MuonChamberMatch *, const reco::MuonSegmentMatch *) override;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonTrackCSCChamberResidual_H

#ifndef Alignment_MuonAlignmentAlgorithms_MuonCSCChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonCSCChamberResidual_H

/** \class MuonCSCChamberResidual
 * 
 * Implementation of muon chamber residuals for CSC
 * 
 * $Id: MuonCSCChamberResidual.h,v 1.3 2011/10/12 23:40:24 khotilov Exp $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonHitsChamberResidual.h"

class MuonCSCChamberResidual: public MuonHitsChamberResidual
{
public:
  MuonCSCChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                         DetId chamberId, AlignableDetOrUnitPtr chamberAlignable);

  // for CSC, the residual is chamber local x, projected by the strip measurement direction
  // for CSC, the resslope is dresx/dz, or tan(phi_y)
  virtual void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit);

  // dummy method
  virtual void setSegmentResidual(const reco::MuonChamberMatch *, const reco::MuonSegmentMatch *) {}
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonCSCChamberResidual_H

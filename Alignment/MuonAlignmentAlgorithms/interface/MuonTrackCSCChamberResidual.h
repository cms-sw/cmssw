#ifndef Alignment_MuonAlignmentAlgorithms_MuonTrackCSCChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonTrackCSCChamberResidual_H

/** \class MuonTrackCSCChamberResidual
 * 
 * Implementation of tracker muon chamber residuals for CSC
 * 
 * $Id: MuonTrackCSCChamberResidual.h,v 1.1 2011/10/12 23:32:07 khotilov Exp $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"

class MuonTrackCSCChamberResidual: public MuonChamberResidual
{
public:
  MuonTrackCSCChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                         DetId chamberId, AlignableDetOrUnitPtr chamberAlignable);
  
  // dummy method
  virtual void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit) {}

  // for CSC, the residual is chamber local x, projected by the strip measurement direction
  // for CSC, the resslope is dresx/dz, or tan(phi_y)
  virtual void setSegmentResidual(const reco::MuonChamberMatch *, const reco::MuonSegmentMatch *);
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonTrackCSCChamberResidual_H

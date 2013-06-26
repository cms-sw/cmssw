#ifndef Alignment_MuonAlignmentAlgorithms_MuonTrackDT2ChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonTrackDT2ChamberResidual_H

/** \class MuonTrackDT2ChamberResidual
 * 
 * Implementation of tracker muon chamber residuals for transverse DT layers
 * 
 * $Id: MuonTrackDT2ChamberResidual.h,v 1.1 2011/10/12 23:32:08 khotilov Exp $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"

class MuonTrackDT2ChamberResidual: public MuonChamberResidual 
{
public:
  MuonTrackDT2ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                              DetId chamberId, AlignableDetOrUnitPtr chamberAlignable);

  // dummy method
  virtual void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit) {}

  // for DT2, the residual is chamber local y
  // for DT2, the resslope is dresy/dz, or tan(phi_x)
  virtual void setSegmentResidual(const reco::MuonChamberMatch *, const reco::MuonSegmentMatch *);
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonTrackDT2ChamberResidual_H

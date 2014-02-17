#ifndef Alignment_MuonAlignmentAlgorithms_MuonDT2ChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonDT2ChamberResidual_H

/** \class MuonDT2ChamberResidual
 * 
 * Implementation of muon chamber residuals for transverse DT layers
 * 
 * $Id: MuonDT2ChamberResidual.h,v 1.3 2011/10/12 23:40:24 khotilov Exp $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonHitsChamberResidual.h"

class MuonDT2ChamberResidual: public MuonHitsChamberResidual 
{
public:
  MuonDT2ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                         DetId chamberId, AlignableDetOrUnitPtr chamberAlignable);
  
  // for DT2, the residual is chamber local y
  // for DT2, the resslope is dresy/dz, or tan(phi_x)
  virtual void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit);

  // dummy method
  virtual void setSegmentResidual(const reco::MuonChamberMatch *, const reco::MuonSegmentMatch *) {}
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonDT2ChamberResidual_H

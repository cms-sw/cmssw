#ifndef Alignment_MuonAlignmentAlgorithms_MuonDT13ChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonDT13ChamberResidual_H

/** \class MuonDT13ChamberResidual
 * 
 * Implementation of muon chamber residuals for axial DT layers
 * 
 * $Id: $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonHitsChamberResidual.h"

class MuonDT13ChamberResidual: public MuonHitsChamberResidual
{
public:
  MuonDT13ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                          DetId chamberId, const AlignableDetOrUnitPtr& chamberAlignable);
  
  // for DT13, the residual is chamber local x
  // for DT13, the resslope is dresx/dz, or tan(phi_y)
  virtual void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit);

  // dummy method
  virtual void setSegmentResidual(const reco::MuonChamberMatch *, const reco::MuonSegmentMatch *) {}
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonDT13ChamberResidual_H

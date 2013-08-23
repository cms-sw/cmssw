#ifndef Alignment_MuonAlignmentAlgorithms_MuonTrackDT13ChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonTrackDT13ChamberResidual_H

/** \class MuonTrackDT13ChamberResidual
 * 
 * Implementation of tracker muon chamber residuals for axial DT layers
 * 
 * $Id: $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"

class MuonTrackDT13ChamberResidual: public MuonChamberResidual
{
public:
  MuonTrackDT13ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                          DetId chamberId, const AlignableDetOrUnitPtr& chamberAlignable);

  // dummy method
  virtual void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit) {}

  // for DT13, the residual is chamber local x
  // for DT13, the resslope is dresx/dz, or tan(phi_y)
  virtual void setSegmentResidual(const reco::MuonChamberMatch *, const reco::MuonSegmentMatch *);
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonTrackDT13ChamberResidual_H

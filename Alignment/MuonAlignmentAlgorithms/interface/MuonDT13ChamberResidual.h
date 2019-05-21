#ifndef Alignment_MuonAlignmentAlgorithms_MuonDT13ChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonDT13ChamberResidual_H

/** \class MuonDT13ChamberResidual
 * 
 * Implementation of muon chamber residuals for axial DT layers
 * 
 * $Id: $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonHitsChamberResidual.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

class MuonDT13ChamberResidual : public MuonHitsChamberResidual {
public:
  MuonDT13ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
                          AlignableNavigator *navigator,
                          DetId chamberId,
                          AlignableDetOrUnitPtr chamberAlignable);

  // for DT13, the residual is chamber local x
  // for DT13, the resslope is dresx/dz, or tan(phi_y)
  void addResidual(edm::ESHandle<Propagator> prop,
                   const TrajectoryStateOnSurface *tsos,
                   const TrackingRecHit *hit,
                   double,
                   double) override;

  // dummy method
  void setSegmentResidual(const reco::MuonChamberMatch *, const reco::MuonSegmentMatch *) override {}
};

#endif  // Alignment_MuonAlignmentAlgorithms_MuonDT13ChamberResidual_H

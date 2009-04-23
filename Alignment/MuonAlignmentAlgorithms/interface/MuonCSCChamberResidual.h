#ifndef Alignment_MuonAlignmentAlgorithms_MuonCSCChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonCSCChamberResidual_H

/** \class MuonCSCChamberResidual
 *  $Date: 2009/02/27 18:58:29 $
 *  $Revision: 1.1 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"

class MuonCSCChamberResidual: public MuonChamberResidual {
public:
  MuonCSCChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator, DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
    : MuonChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable)
  {};

  int type() const { return MuonChamberResidual::kCSC; };

  // for CSC, the residual is chamber local x, projected by the strip measurement direction
  // for CSC, the resslope is dresx/dz, or tan(phi_y)
  void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit);

  double signConvention(const unsigned int rawId=0) const {
    DetId id = m_chamberId;
    if (rawId != 0) id = DetId(rawId);
    GlobalVector zDirection(0., 0., 1.);
    return (m_globalGeometry->idToDet(id)->toLocal(zDirection).z() > 0. ? 1. : -1.);
  };
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonCSCChamberResidual_H

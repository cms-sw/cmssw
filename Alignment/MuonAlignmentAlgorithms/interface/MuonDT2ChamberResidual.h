#ifndef Alignment_MuonAlignmentAlgorithms_MuonDT2ChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonDT2ChamberResidual_H

/** \class MuonDT2ChamberResidual
 *  $Date: 2009/02/27 18:58:29 $
 *  $Revision: 1.1 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

class MuonDT2ChamberResidual: public MuonChamberResidual {
public:
  MuonDT2ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator, DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
    : MuonChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable)
  {};

  int type() const { return MuonChamberResidual::kDT2; };

  // for DT2, the residual is chamber local y
  // for DT2, the resslope is dresy/dz, or tan(phi_x)
  void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit);

  // for DT2, the global direction is CMS z (e.g. local y without the sign convention issues)
  double signConvention(const unsigned int rawId=0) const {
    DetId id = m_chamberId;
    if (rawId != 0) id = DetId(rawId);
    GlobalVector zDirection(0., 0., 1.);
    return (m_globalGeometry->idToDet(id)->toLocal(zDirection).y() > 0. ? 1. : -1.);
  };
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonDT2ChamberResidual_H

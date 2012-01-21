#ifndef Alignment_MuonAlignmentAlgorithms_MuonDT13ChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonDT13ChamberResidual_H

/** \class MuonDT13ChamberResidual
 *  $Date: 2009/02/27 18:58:29 $
 *  $Revision: 1.1 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

class MuonDT13ChamberResidual: public MuonChamberResidual {
public:
  MuonDT13ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator, DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
    : MuonChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable)
  {};

  int type() const { return MuonChamberResidual::kDT13; };

  // for DT13, the residual is chamber local x
  // for DT13, the resslope is dresx/dz, or tan(phi_y)
  void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit);

  // for DT13, the global direction is a polygonal rphi (e.g. local x without the sign convention issues)
  double signConvention(const unsigned int rawId=0) const {
    DetId id = m_chamberId;
    if (rawId != 0) id = DetId(rawId);
    double rphiAngle = atan2(m_globalGeometry->idToDet(id)->position().y(), m_globalGeometry->idToDet(id)->position().x()) + M_PI/2.;
    GlobalVector rphiDirection(cos(rphiAngle), sin(rphiAngle), 0.);
    return (m_globalGeometry->idToDet(id)->toLocal(rphiDirection).x() > 0. ? 1. : -1.);
  };
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonDT13ChamberResidual_H

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonCSCChamberResidual.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

void MuonCSCChamberResidual::addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit) {
  DetId id = hit->geographicalId();
  const CSCGeometry *cscGeometry = dynamic_cast<const CSCGeometry*>(m_globalGeometry->slaveGeometry(id));
  assert(cscGeometry);

  LocalPoint hitChamberPos = m_chamberAlignable->surface().toLocal(m_globalGeometry->idToDet(id)->toGlobal(hit->localPosition()));
  LocalPoint tsosChamberPos = m_chamberAlignable->surface().toLocal(m_globalGeometry->idToDet(id)->toGlobal(tsos->localPosition()));

  int strip = cscGeometry->layer(id)->geometry()->nearestStrip(hit->localPosition());
  double angle = cscGeometry->layer(id)->geometry()->stripAngle(strip) - M_PI/2.;
  double sinAngle = sin(angle);
  double cosAngle = cos(angle);

  double residual = cosAngle * (tsosChamberPos.x() - hitChamberPos.x()) + sinAngle * (tsosChamberPos.y() - hitChamberPos.y());  // yes, that's +sin()

  double xx = hit->localPositionError().xx();
  double xy = hit->localPositionError().xy();
  double yy = hit->localPositionError().yy();
  double weight = 1. / (xx*cosAngle*cosAngle + 2.*xy*sinAngle*cosAngle + yy*sinAngle*sinAngle);

  double layerPosition = tsosChamberPos.z();  // the layer's position in the chamber's coordinate system

  m_numHits++;

  // "x" is the layerPosition, "y" is the residual (this is a linear fit to residual versus layerPosition)
  m_residual_1 += weight;
  m_residual_x += weight * layerPosition;
  m_residual_y += weight * residual;
  m_residual_xx += weight * layerPosition * layerPosition;
  m_residual_xy += weight * layerPosition * residual;

  // "x" is the layerPosition, "y" is chamberx (this is a linear fit to chamberx versus layerPosition)
  m_trackx_1 += weight;
  m_trackx_x += weight * layerPosition;
  m_trackx_y += weight * tsosChamberPos.x();
  m_trackx_xx += weight * layerPosition * layerPosition;
  m_trackx_xy += weight * layerPosition * tsosChamberPos.x();

  // "x" is the layerPosition, "y" is chambery (this is a linear fit to chambery versus layerPosition)
  m_tracky_1 += weight;
  m_tracky_x += weight * layerPosition;
  m_tracky_y += weight * tsosChamberPos.y();
  m_tracky_xx += weight * layerPosition * layerPosition;
  m_tracky_xy += weight * layerPosition * tsosChamberPos.y();

  m_localIDs.push_back(id);
  m_localResids.push_back(residual);
  m_individual_x.push_back(layerPosition);
  m_individual_y.push_back(residual);
  m_individual_weight.push_back(weight);
}

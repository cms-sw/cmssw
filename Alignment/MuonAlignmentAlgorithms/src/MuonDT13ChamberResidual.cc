/*
 * $Id: $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonDT13ChamberResidual.h"


MuonDT13ChamberResidual::MuonDT13ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                                                 DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
  : MuonHitsChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable)
{
  m_type = MuonChamberResidual::kDT13;
  double rphiAngle = atan2(m_globalGeometry->idToDet(m_chamberId)->position().y(), m_globalGeometry->idToDet(m_chamberId)->position().x()) + M_PI/2.;
  align::GlobalVector rphiDirection(cos(rphiAngle), sin(rphiAngle), 0.);
  m_sign = m_globalGeometry->idToDet(m_chamberId)->toLocal(rphiDirection).x() > 0. ? 1. : -1.;
}


void MuonDT13ChamberResidual::addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit)
{
  DetId id = hit->geographicalId();

  align::LocalPoint hitChamberPos  = m_chamberAlignable->surface().toLocal(m_globalGeometry->idToDet(id)->toGlobal(hit->localPosition()));
  align::LocalPoint tsosChamberPos = m_chamberAlignable->surface().toLocal(m_globalGeometry->idToDet(id)->toGlobal(tsos->localPosition()));

  double residual = tsosChamberPos.x() - hitChamberPos.x();  // residual is hit minus hit
  double weight = 1. / hit->localPositionError().xx();  // weight linear fit by hit-only local error
  double layerPosition = tsosChamberPos.z();  // the layer's position in the chamber's coordinate system
  double layerHitPos = hitChamberPos.z();

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

  m_hitx_1 += weight;
  m_hitx_x += weight * layerHitPos;
  m_hitx_y += weight * hitChamberPos.x();
  m_hitx_xx += weight * layerHitPos * layerHitPos;
  m_hitx_xy += weight * layerHitPos * hitChamberPos.x();
  
  m_hity_1 += weight;
  m_hity_x += weight * layerHitPos;
  m_hity_y += weight * hitChamberPos.y();
  m_hity_xx += weight * layerHitPos * layerPosition;
  m_hity_xy += weight * layerHitPos * hitChamberPos.y();

  m_localIDs.push_back(id);
  m_localResids.push_back(tsos->localPosition().x() - hit->localPosition().x());
  m_individual_x.push_back(layerPosition);
  m_individual_y.push_back(residual);
  m_individual_weight.push_back(weight);

  if (m_numHits>1) segment_fit();
}

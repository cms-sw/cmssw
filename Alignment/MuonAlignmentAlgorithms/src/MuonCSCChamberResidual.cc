/* 
 * $Id: $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonCSCChamberResidual.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

MuonCSCChamberResidual::MuonCSCChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
                                               AlignableNavigator *navigator,
                                               DetId chamberId,
                                               AlignableDetOrUnitPtr chamberAlignable)
    : MuonHitsChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable) {
  m_type = MuonChamberResidual::kCSC;
  align::GlobalVector zDirection(0., 0., 1.);
  m_sign = m_globalGeometry->idToDet(m_chamberId)->toLocal(zDirection).z() > 0. ? 1. : -1.;
}

void MuonCSCChamberResidual::addResidual(edm::ESHandle<Propagator> prop,
                                         const TrajectoryStateOnSurface *tsos,
                                         const TrackingRecHit *hit,
                                         double chamber_width,
                                         double chamber_length) {
  bool m_debug = false;

  if (m_debug)
    std::cout << "MuonCSCChamberResidual::addResidual 1" << std::endl;
  DetId id = hit->geographicalId();
  if (m_debug)
    std::cout << "MuonCSCChamberResidual::addResidual 2" << std::endl;
  const CSCGeometry *cscGeometry = dynamic_cast<const CSCGeometry *>(m_globalGeometry->slaveGeometry(id));
  if (m_debug)
    std::cout << "MuonCSCChamberResidual::addResidual 3" << std::endl;
  assert(cscGeometry);

  if (m_debug) {
    std::cout << " MuonCSCChamberResidual hit->localPosition() x: " << hit->localPosition().x()
              << " tsos->localPosition() x: " << tsos->localPosition().x() << std::endl;
    std::cout << "                        hit->localPosition() y: " << hit->localPosition().y()
              << " tsos->localPosition() y: " << tsos->localPosition().y() << std::endl;
    std::cout << "                        hit->localPosition() z: " << hit->localPosition().z()
              << " tsos->localPosition() z: " << tsos->localPosition().z() << std::endl;
  }

  // hit->localPosition() is coordinate in local system of LAYER. Transfer it to coordiante in local system of chamber
  align::LocalPoint hitChamberPos =
      m_chamberAlignable->surface().toLocal(m_globalGeometry->idToDet(id)->toGlobal(hit->localPosition()));
  // TSOS->localPosition() is given in local system of CHAMBER (for segment-based reconstruction)
  //  align::LocalPoint tsosChamberPos = tsos->localPosition();
  align::LocalPoint tsosChamberPos =
      m_chamberAlignable->surface().toLocal(m_globalGeometry->idToDet(id)->toGlobal(tsos->localPosition()));

  int strip = cscGeometry->layer(id)->geometry()->nearestStrip(hit->localPosition());
  double angle = cscGeometry->layer(id)->geometry()->stripAngle(strip) - M_PI / 2.;
  double sinAngle = sin(angle);
  double cosAngle = cos(angle);

  double residual = cosAngle * (tsosChamberPos.x() - hitChamberPos.x()) +
                    sinAngle * (tsosChamberPos.y() - hitChamberPos.y());  // yes, that's +sin()

  if (m_debug)
    std::cout << " MuonCSCChamberResidual residual: " << residual << std::endl;

  double xx = hit->localPositionError().xx();
  double xy = hit->localPositionError().xy();
  double yy = hit->localPositionError().yy();
  double weight = 1. / (xx * cosAngle * cosAngle + 2. * xy * sinAngle * cosAngle + yy * sinAngle * sinAngle);

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
  m_hity_xx += weight * layerHitPos * layerHitPos;
  m_hity_xy += weight * layerHitPos * hitChamberPos.y();

  m_localIDs.push_back(id);
  m_localResids.push_back(residual);  // FIXME Check if this method is needed
  m_individual_x.push_back(layerPosition);
  m_individual_y.push_back(residual);
  m_individual_weight.push_back(weight);

  if (m_numHits > 1)
    segment_fit();
}

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonDTChamberResidual.h"

void MuonDTChamberResidual::addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit) {
  DetId id = hit->geographicalId();

  double residual = tsos->localPosition().x() - hit->localPosition().x();  // residual is track minus hit
  double weight = 1. / hit->localPositionError().xx();
  
  LocalPoint localPos = tsos->localPosition();  // the "position" is from the projected track, not the hit
  GlobalPoint globalPos = m_globalGeometry->idToDet(id)->toGlobal(localPos);
  double phi = globalPos.phi();
  double z = globalPos.z();  
  double R = globalPos.perp();
  LocalPoint chamberPos = m_globalGeometry->idToDet(m_chamberId)->toLocal(globalPos);
  
  GlobalVector globalResid = m_globalGeometry->idToDet(id)->toGlobal(LocalVector(residual, 0., 0.));

  AlgebraicMatrix derivatives = m_chamberAlignable->alignmentParameters()->derivatives(*tsos, m_navigator->alignableFromDetId(id));
  derivatives = derivatives.sub(1, 6, 1, 1);  // just the measured part
  AlgebraicVector residual_vector(1);
  residual_vector[0] = residual;
  AlgebraicVector alignment_residuals = derivatives * residual_vector;

  if (DTSuperLayerId(id.rawId()).superLayer() == 2) {
    m_SL2_N++;
    m_SL2_denom += weight;
    m_SL2_residglobal += globalResid.z() * weight;
    m_SL2_residy += alignment_residuals[1] * weight;
    m_SL2_zpos += z * weight;
    m_SL2_Rpos += R * weight;
    m_SL2_phipos += phi * weight;
    if (m_SL2_N == 1  ||  phi < m_SL2_phimin) m_SL2_phimin = phi;
    if (m_SL2_N == 1  ||  phi > m_SL2_phimax) m_SL2_phimax = phi;
    m_SL2_localxpos += chamberPos.x() * weight;
    m_SL2_localypos += chamberPos.y() * weight;

    // Note: HIP algorithm does not weight alignment parameters by their derivatives
    // (the derivative and 1/derivative cancel in the similarity transform
    // when the covariance matrix is diagonal)

//     double residz_weight = weight; // * fabs(derivatives[2][0]);
//     m_residz += alignment_residuals[2] * residz_weight;
//     m_residz_denom += residz_weight;

    double residphix_weight = weight; // * fabs(derivatives[3][0]);
    m_residphix += alignment_residuals[3] * residphix_weight;
    m_residphix_denom += residphix_weight;

//     double residphiz_weight = weight; // * fabs(derivatives[5][0]);
//     m_residphiz += alignment_residuals[5] * residphiz_weight;
//     m_residphiz_denom += residphiz_weight;
  }

  else { // superLayer 1 or 3
    GlobalVector globalRphiDirection = m_globalGeometry->idToDet(id)->toGlobal(LocalVector(1., 0., 0.));
    if ((GlobalVector(globalPos.x(), globalPos.y(), 0.).cross(globalRphiDirection)).z() < 0.) {
      globalRphiDirection = -globalRphiDirection;
    }

    m_SL13_N++;
    m_SL13_denom += weight;
    m_SL13_residglobal += globalResid.dot(globalRphiDirection) * weight;
    m_SL13_residx += alignment_residuals[0] * weight;
    m_SL13_zpos += z * weight;
    m_SL13_Rpos += R * weight;
    m_SL13_phipos += phi * weight;
    if (m_SL13_N == 1  ||  phi < m_SL13_phimin) m_SL13_phimin = phi;
    if (m_SL13_N == 1  ||  phi > m_SL13_phimax) m_SL13_phimax = phi;
    m_SL13_localxpos += chamberPos.x() * weight;
    m_SL13_localypos += chamberPos.y() * weight;

    double residz_weight = weight; // * fabs(derivatives[2][0]);
    m_residz += alignment_residuals[2] * residz_weight;
    m_residz_denom += residz_weight;

    double residphiy_weight = weight; // * fabs(derivatives[4][0]);
    m_residphiy += alignment_residuals[4] * residphiy_weight;
    m_residphiy_denom += residphiy_weight;

    double residphiz_weight = weight; // * fabs(derivatives[5][0]);
    m_residphiz += alignment_residuals[5] * residphiz_weight;
    m_residphiz_denom += residphiz_weight;
  } // end which superLayer
}

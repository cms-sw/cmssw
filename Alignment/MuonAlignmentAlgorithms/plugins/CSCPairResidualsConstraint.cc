#include <iomanip>
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Alignment/MuonAlignmentAlgorithms/plugins/CSCOverlapsAlignmentAlgorithm.h"
#include "Alignment/MuonAlignmentAlgorithms/plugins/CSCPairResidualsConstraint.h"

double CSCPairResidualsConstraint::value() const {
  double delta = (m_sum1 * m_sumxx) - (m_sumx * m_sumx);
  assert(delta > 0.);
  if (m_parent->m_mode == kModePhiy || m_parent->m_mode == kModePhiPos || m_parent->m_mode == kModeRadius) {
    return ((m_sumxx * m_sumy) - (m_sumx * m_sumxy)) / delta;
  } else if (m_parent->m_mode == kModePhiz) {
    return ((m_sum1 * m_sumxy) - (m_sumx * m_sumy)) / delta;
  } else
    assert(false);
}

double CSCPairResidualsConstraint::error() const {
  if (m_parent->m_errorFromRMS) {
    assert(m_sum1 > 0.);
    return sqrt((m_sumyy / m_sum1) - pow(m_sumy / m_sum1, 2)) / sqrt(m_sumN);
  } else {
    double delta = (m_sum1 * m_sumxx) - (m_sumx * m_sumx);
    assert(delta > 0.);
    if (m_parent->m_mode == kModePhiy || m_parent->m_mode == kModePhiPos || m_parent->m_mode == kModeRadius) {
      return sqrt(m_sumxx / delta);
    } else if (m_parent->m_mode == kModePhiz) {
      return sqrt(m_sum1 / delta);
    } else
      assert(false);
  }
}

bool CSCPairResidualsConstraint::valid() const { return (m_sumN >= m_parent->m_minTracksPerOverlap); }

void CSCPairResidualsConstraint::configure(CSCOverlapsAlignmentAlgorithm *parent) {
  m_parent = parent;

  if (m_parent->m_makeHistograms) {
    edm::Service<TFileService> tFileService;

    std::stringstream name, name2, name3, title;
    title << "i =" << m_id_i << " j =" << m_id_j;

    name << "slopeResiduals_" << m_identifier;
    m_slopeResiduals = tFileService->make<TH1F>(name.str().c_str(), title.str().c_str(), 300, -30., 30.);

    name2 << "offsetResiduals_" << m_identifier;
    m_offsetResiduals = tFileService->make<TH1F>(name2.str().c_str(), title.str().c_str(), 300, -30., 30.);

    name3 << "radial_" << m_identifier;
    m_radial = tFileService->make<TH1F>(name3.str().c_str(), title.str().c_str(), 700, 0., 700.);
  } else {
    m_slopeResiduals = nullptr;
    m_offsetResiduals = nullptr;
    m_radial = nullptr;
  }
}

void CSCPairResidualsConstraint::setZplane(const CSCGeometry *cscGeometry) {
  m_cscGeometry = cscGeometry;

  m_Zplane = (m_cscGeometry->idToDet(m_id_i)->surface().position().z() +
              m_cscGeometry->idToDet(m_id_j)->surface().position().z()) /
             2.;
  m_averageRadius = (m_cscGeometry->idToDet(m_id_i)->surface().position().perp() +
                     m_cscGeometry->idToDet(m_id_j)->surface().position().perp()) /
                    2.;

  m_iZ = m_cscGeometry->idToDet(m_id_i)->surface().position().z();
  m_jZ = m_cscGeometry->idToDet(m_id_j)->surface().position().z();

  CSCDetId i1(m_id_i.endcap(), m_id_i.station(), m_id_i.ring(), m_id_i.chamber(), 1);
  CSCDetId i6(m_id_i.endcap(), m_id_i.station(), m_id_i.ring(), m_id_i.chamber(), 6);
  CSCDetId j1(m_id_j.endcap(), m_id_j.station(), m_id_j.ring(), m_id_j.chamber(), 1);
  CSCDetId j6(m_id_j.endcap(), m_id_j.station(), m_id_j.ring(), m_id_j.chamber(), 6);
  m_iZ1 = m_cscGeometry->idToDet(m_id_i)->surface().toLocal(m_cscGeometry->idToDet(i1)->surface().position()).z();
  m_iZ6 = m_cscGeometry->idToDet(m_id_i)->surface().toLocal(m_cscGeometry->idToDet(i6)->surface().position()).z();
  m_jZ1 = m_cscGeometry->idToDet(m_id_j)->surface().toLocal(m_cscGeometry->idToDet(j1)->surface().position()).z();
  m_jZ6 = m_cscGeometry->idToDet(m_id_j)->surface().toLocal(m_cscGeometry->idToDet(j6)->surface().position()).z();

  m_Zsurface = Plane::build(Plane::PositionType(0., 0., m_Zplane), Plane::RotationType());
}

void CSCPairResidualsConstraint::setPropagator(const Propagator *propagator) { m_propagator = propagator; }

bool CSCPairResidualsConstraint::addTrack(const std::vector<TrajectoryMeasurement> &measurements,
                                          const reco::TransientTrack &track,
                                          const TrackTransformer *trackTransformer) {
  std::vector<const TransientTrackingRecHit *> hits_i;
  std::vector<const TransientTrackingRecHit *> hits_j;

  for (std::vector<TrajectoryMeasurement>::const_iterator measurement = measurements.begin();
       measurement != measurements.end();
       ++measurement) {
    const TransientTrackingRecHit *hit = &*(measurement->recHit());

    DetId id = hit->geographicalId();
    if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
      CSCDetId cscid(id.rawId());
      CSCDetId chamberId(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber(), 0);
      if (m_parent->m_combineME11 && cscid.station() == 1 && cscid.ring() == 4)
        chamberId = CSCDetId(cscid.endcap(), 1, 1, cscid.chamber(), 0);

      if (chamberId == m_id_i)
        hits_i.push_back(hit);
      if (chamberId == m_id_j)
        hits_j.push_back(hit);
    }
  }

  if (m_parent->m_makeHistograms) {
    m_parent->m_hitsPerChamber->Fill(hits_i.size());
    m_parent->m_hitsPerChamber->Fill(hits_j.size());
  }

  // require minimum number of hits (if the requirement is too low (~2), some NANs might result...)
  if (int(hits_i.size()) < m_parent->m_minHitsPerChamber || int(hits_j.size()) < m_parent->m_minHitsPerChamber)
    return false;

  // maybe require segments to be fiducial
  if (m_parent->m_fiducial && !(isFiducial(hits_i, true) && isFiducial(hits_j, false)))
    return false;

  double intercept_i = 0.;
  double interceptError2_i = 0.;
  double slope_i = 0.;
  double slopeError2_i = 0.;
  double intercept_j = 0.;
  double interceptError2_j = 0.;
  double slope_j = 0.;
  double slopeError2_j = 0.;

  // if slopeFromTrackRefit, then you'll need to refit the whole track without this station's hits;
  // need at least two other stations for that to be reliable
  if (m_parent->m_slopeFromTrackRefit) {
    double dphidz;
    if (dphidzFromTrack(measurements, track, trackTransformer, dphidz)) {
      double sum1_i = 0.;
      double sumy_i = 0.;
      for (std::vector<const TransientTrackingRecHit *>::const_iterator hit = hits_i.begin(); hit != hits_i.end();
           ++hit) {
        double phi, phierr2;
        calculatePhi(*hit, phi, phierr2, false, true);
        double z = (*hit)->globalPosition().z() - m_Zplane;

        double weight = 1.;
        if (m_parent->m_useHitWeights)
          weight = 1. / phierr2;
        sum1_i += weight;
        sumy_i += weight * (phi - z * dphidz);
      }

      double sum1_j = 0.;
      double sumy_j = 0.;
      for (std::vector<const TransientTrackingRecHit *>::const_iterator hit = hits_j.begin(); hit != hits_j.end();
           ++hit) {
        double phi, phierr2;
        calculatePhi(*hit, phi, phierr2, false, true);
        double z = (*hit)->globalPosition().z() - m_Zplane;

        double weight = 1.;
        if (m_parent->m_useHitWeights)
          weight = 1. / phierr2;
        sum1_j += weight;
        sumy_j += weight * (phi - z * dphidz);
      }

      if (sum1_i != 0. && sum1_j != 0.) {
        slope_i = slope_j = dphidz;

        intercept_i = sumy_i / sum1_i;
        interceptError2_i = 1. / sum1_i;

        intercept_j = sumy_j / sum1_j;
        interceptError2_j = 1. / sum1_j;
      } else
        return false;
    }
  }

  else {  // not slopeFromTrackRefit
    double sum1_i = 0.;
    double sumx_i = 0.;
    double sumy_i = 0.;
    double sumxx_i = 0.;
    double sumxy_i = 0.;
    for (std::vector<const TransientTrackingRecHit *>::const_iterator hit = hits_i.begin(); hit != hits_i.end();
         ++hit) {
      double phi, phierr2;
      calculatePhi(*hit, phi, phierr2, false, true);
      double z = (*hit)->globalPosition().z() - m_Zplane;

      double weight = 1.;
      if (m_parent->m_useHitWeights)
        weight = 1. / phierr2;
      sum1_i += weight;
      sumx_i += weight * z;
      sumy_i += weight * phi;
      sumxx_i += weight * z * z;
      sumxy_i += weight * z * phi;
    }

    double sum1_j = 0.;
    double sumx_j = 0.;
    double sumy_j = 0.;
    double sumxx_j = 0.;
    double sumxy_j = 0.;
    for (std::vector<const TransientTrackingRecHit *>::const_iterator hit = hits_j.begin(); hit != hits_j.end();
         ++hit) {
      double phi, phierr2;
      calculatePhi(*hit, phi, phierr2, false, true);
      double z = (*hit)->globalPosition().z() - m_Zplane;

      double weight = 1.;
      if (m_parent->m_useHitWeights)
        weight = 1. / phierr2;
      sum1_j += weight;
      sumx_j += weight * z;
      sumy_j += weight * phi;
      sumxx_j += weight * z * z;
      sumxy_j += weight * z * phi;
    }

    double delta_i = (sum1_i * sumxx_i) - (sumx_i * sumx_i);
    double delta_j = (sum1_j * sumxx_j) - (sumx_j * sumx_j);
    if (delta_i != 0. && delta_j != 0.) {
      intercept_i = ((sumxx_i * sumy_i) - (sumx_i * sumxy_i)) / delta_i;
      interceptError2_i = sumxx_i / delta_i;
      slope_i = ((sum1_i * sumxy_i) - (sumx_i * sumy_i)) / delta_i;
      slopeError2_i = sum1_i / delta_i;

      intercept_j = ((sumxx_j * sumy_j) - (sumx_j * sumxy_j)) / delta_j;
      interceptError2_j = sumxx_j / delta_j;
      slope_j = ((sum1_j * sumxy_j) - (sumx_j * sumy_j)) / delta_j;
      slopeError2_j = sum1_j / delta_j;
    } else
      return false;
  }

  // from hits on the two chambers, determine radial_intercepts separately and radial_slope together
  double sum1_ri = 0.;
  double sumx_ri = 0.;
  double sumy_ri = 0.;
  double sumxx_ri = 0.;
  double sumxy_ri = 0.;
  for (std::vector<const TransientTrackingRecHit *>::const_iterator hit = hits_i.begin(); hit != hits_i.end(); ++hit) {
    double r = (*hit)->globalPosition().perp();
    double z = (*hit)->globalPosition().z() - m_Zplane;
    sum1_ri += 1.;
    sumx_ri += z;
    sumy_ri += r;
    sumxx_ri += z * z;
    sumxy_ri += z * r;
  }
  double radial_delta_i = (sum1_ri * sumxx_ri) - (sumx_ri * sumx_ri);
  if (radial_delta_i == 0.)
    return false;
  double radial_slope_i = ((sum1_ri * sumxy_ri) - (sumx_ri * sumy_ri)) / radial_delta_i;
  double radial_intercept_i =
      ((sumxx_ri * sumy_ri) - (sumx_ri * sumxy_ri)) / radial_delta_i + radial_slope_i * (m_iZ - m_Zplane);

  double sum1_rj = 0.;
  double sumx_rj = 0.;
  double sumy_rj = 0.;
  double sumxx_rj = 0.;
  double sumxy_rj = 0.;
  for (std::vector<const TransientTrackingRecHit *>::const_iterator hit = hits_j.begin(); hit != hits_j.end(); ++hit) {
    double r = (*hit)->globalPosition().perp();
    double z = (*hit)->globalPosition().z() - m_Zplane;
    sum1_rj += 1.;
    sumx_rj += z;
    sumy_rj += r;
    sumxx_rj += z * z;
    sumxy_rj += z * r;
  }
  double radial_delta_j = (sum1_rj * sumxx_rj) - (sumx_rj * sumx_rj);
  if (radial_delta_j == 0.)
    return false;
  double radial_slope_j = ((sum1_rj * sumxy_rj) - (sumx_rj * sumy_rj)) / radial_delta_j;
  double radial_intercept_j =
      ((sumxx_rj * sumy_rj) - (sumx_rj * sumxy_rj)) / radial_delta_j + radial_slope_j * (m_jZ - m_Zplane);

  double radial_delta = ((sum1_ri + sum1_rj) * (sumxx_ri + sumxx_rj)) - ((sumx_ri + sumx_rj) * (sumx_ri + sumx_rj));
  if (radial_delta == 0.)
    return false;
  double radial_intercept =
      (((sumxx_ri + sumxx_rj) * (sumy_ri + sumy_rj)) - ((sumx_ri + sumx_rj) * (sumxy_ri + sumxy_rj))) / radial_delta;
  double radial_slope =
      (((sum1_ri + sum1_rj) * (sumxy_ri + sumxy_rj)) - ((sumx_ri + sumx_rj) * (sumy_ri + sumy_rj))) / radial_delta;

  if (m_parent->m_makeHistograms) {
    m_parent->m_drdz->Fill(radial_slope);
  }
  if (m_parent->m_maxdrdz > 0. && fabs(radial_slope) > m_parent->m_maxdrdz)
    return false;

  double quantity = 0.;
  double quantityError2 = 0.;
  if (m_parent->m_mode == kModePhiy) {  // phiy comes from track d(rphi)/dz
    quantity = (slope_i * radial_intercept_i) - (slope_j * radial_intercept_j);
    quantityError2 = (slopeError2_i)*pow(radial_intercept_i, 2) + (slopeError2_j)*pow(radial_intercept_j, 2);
  } else if (m_parent->m_mode == kModePhiPos || m_parent->m_mode == kModeRadius) {  // phipos comes from phi intercepts
    quantity = intercept_i - intercept_j;
    quantityError2 = interceptError2_i + interceptError2_j;
  } else if (m_parent->m_mode == kModePhiz) {  // phiz comes from the slope of rphi intercepts
    quantity = (intercept_i - intercept_j) * radial_intercept;
    quantityError2 = (interceptError2_i + interceptError2_j) * pow(radial_intercept, 2);
  } else
    assert(false);

  if (quantityError2 == 0.)
    return false;

  double slopeResid = ((slope_i * radial_intercept_i) - (slope_j * radial_intercept_j)) * 1000.;
  double slopeResidError2 =
      ((slopeError2_i)*pow(radial_intercept_i, 2) + (slopeError2_j)*pow(radial_intercept_j, 2)) * 1000. * 1000.;
  double offsetResid = (intercept_i - intercept_j) * radial_intercept * 10.;
  double offsetResidError2 = (interceptError2_i + interceptError2_j) * pow(radial_intercept, 2) * 10. * 10.;

  if (m_parent->m_truncateSlopeResid > 0. && fabs(slopeResid) > m_parent->m_truncateSlopeResid)
    return false;
  if (m_parent->m_truncateOffsetResid > 0. && fabs(offsetResid) > m_parent->m_truncateOffsetResid)
    return false;

  double weight = 1.;
  if (m_parent->m_useTrackWeights)
    weight = 1. / quantityError2;

  // fill the running sums for this CSCPairResidualsConstraint
  m_sumN += 1;
  m_sum1 += weight;
  m_sumx += weight * (radial_intercept - m_averageRadius);
  m_sumy += weight * quantity;
  m_sumxx += weight * pow(radial_intercept - m_averageRadius, 2);
  m_sumyy += weight * quantity * quantity;
  m_sumxy += weight * (radial_intercept - m_averageRadius) * quantity;

  if (m_parent->m_makeHistograms) {
    double rphi_slope_i = slope_i * radial_intercept_i;
    double rphi_slope_j = slope_j * radial_intercept_j;

    if (m_parent->m_slopeFromTrackRefit) {
      m_parent->m_slope->Fill(rphi_slope_i);  // == rphi_slope_j

      if (m_id_i.endcap() == 1 && m_id_i.station() == 4)
        m_parent->m_slope_MEp4->Fill(rphi_slope_i);
      if (m_id_i.endcap() == 1 && m_id_i.station() == 3)
        m_parent->m_slope_MEp3->Fill(rphi_slope_i);
      if (m_id_i.endcap() == 1 && m_id_i.station() == 2)
        m_parent->m_slope_MEp2->Fill(rphi_slope_i);
      if (m_id_i.endcap() == 1 && m_id_i.station() == 1)
        m_parent->m_slope_MEp1->Fill(rphi_slope_i);
      if (m_id_i.endcap() == 2 && m_id_i.station() == 1)
        m_parent->m_slope_MEm1->Fill(rphi_slope_i);
      if (m_id_i.endcap() == 2 && m_id_i.station() == 2)
        m_parent->m_slope_MEm2->Fill(rphi_slope_i);
      if (m_id_i.endcap() == 2 && m_id_i.station() == 3)
        m_parent->m_slope_MEm3->Fill(rphi_slope_i);
      if (m_id_i.endcap() == 2 && m_id_i.station() == 4)
        m_parent->m_slope_MEm4->Fill(rphi_slope_i);
    } else {
      m_parent->m_slope->Fill(rphi_slope_i);
      m_parent->m_slope->Fill(rphi_slope_j);

      if (m_id_i.endcap() == 1 && m_id_i.station() == 4) {
        m_parent->m_slope_MEp4->Fill(rphi_slope_i);
        m_parent->m_slope_MEp4->Fill(rphi_slope_j);
      }
      if (m_id_i.endcap() == 1 && m_id_i.station() == 3) {
        m_parent->m_slope_MEp3->Fill(rphi_slope_i);
        m_parent->m_slope_MEp3->Fill(rphi_slope_j);
      }
      if (m_id_i.endcap() == 1 && m_id_i.station() == 2) {
        m_parent->m_slope_MEp2->Fill(rphi_slope_i);
        m_parent->m_slope_MEp2->Fill(rphi_slope_j);
      }
      if (m_id_i.endcap() == 1 && m_id_i.station() == 1) {
        m_parent->m_slope_MEp1->Fill(rphi_slope_i);
        m_parent->m_slope_MEp1->Fill(rphi_slope_j);
      }
      if (m_id_i.endcap() == 2 && m_id_i.station() == 1) {
        m_parent->m_slope_MEm1->Fill(rphi_slope_i);
        m_parent->m_slope_MEm1->Fill(rphi_slope_j);
      }
      if (m_id_i.endcap() == 2 && m_id_i.station() == 2) {
        m_parent->m_slope_MEm2->Fill(rphi_slope_i);
        m_parent->m_slope_MEm2->Fill(rphi_slope_j);
      }
      if (m_id_i.endcap() == 2 && m_id_i.station() == 3) {
        m_parent->m_slope_MEm3->Fill(rphi_slope_i);
        m_parent->m_slope_MEm3->Fill(rphi_slope_j);
      }
      if (m_id_i.endcap() == 2 && m_id_i.station() == 4) {
        m_parent->m_slope_MEm4->Fill(rphi_slope_i);
        m_parent->m_slope_MEm4->Fill(rphi_slope_j);
      }
    }

    m_slopeResiduals->Fill(slopeResid);
    m_offsetResiduals->Fill(offsetResid);
    m_radial->Fill(radial_intercept);

    m_parent->m_slopeResiduals->Fill(slopeResid);
    m_parent->m_slopeResiduals_weighted->Fill(slopeResid, 1. / slopeResidError2);
    m_parent->m_slopeResiduals_normalized->Fill(slopeResid / sqrt(slopeResidError2));

    m_parent->m_offsetResiduals->Fill(offsetResid);
    m_parent->m_offsetResiduals_weighted->Fill(offsetResid, 1. / offsetResidError2);
    m_parent->m_offsetResiduals_normalized->Fill(offsetResid / sqrt(offsetResidError2));

    double ringbin = 0;
    if (m_id_i.endcap() == 2 && m_id_i.station() == 4 && m_id_i.ring() == 2)
      ringbin = 1.5;
    else if (m_id_i.endcap() == 2 && m_id_i.station() == 4 && m_id_i.ring() == 1)
      ringbin = 2.5;
    else if (m_id_i.endcap() == 2 && m_id_i.station() == 3 && m_id_i.ring() == 2)
      ringbin = 3.5;
    else if (m_id_i.endcap() == 2 && m_id_i.station() == 3 && m_id_i.ring() == 1)
      ringbin = 4.5;
    else if (m_id_i.endcap() == 2 && m_id_i.station() == 2 && m_id_i.ring() == 2)
      ringbin = 5.5;
    else if (m_id_i.endcap() == 2 && m_id_i.station() == 2 && m_id_i.ring() == 1)
      ringbin = 6.5;
    else if (m_id_i.endcap() == 2 && m_id_i.station() == 1 && m_id_i.ring() == 3)
      ringbin = 7.5;
    else if (m_id_i.endcap() == 2 && m_id_i.station() == 1 && m_id_i.ring() == 2)
      ringbin = 8.5;
    else if (m_id_i.endcap() == 2 && m_id_i.station() == 1 && m_id_i.ring() == 1)
      ringbin = 9.5;
    else if (m_id_i.endcap() == 2 && m_id_i.station() == 1 && m_id_i.ring() == 4)
      ringbin = 10.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 1 && m_id_i.ring() == 4)
      ringbin = 11.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 1 && m_id_i.ring() == 1)
      ringbin = 12.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 1 && m_id_i.ring() == 2)
      ringbin = 13.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 1 && m_id_i.ring() == 3)
      ringbin = 14.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 2 && m_id_i.ring() == 1)
      ringbin = 15.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 2 && m_id_i.ring() == 2)
      ringbin = 16.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 3 && m_id_i.ring() == 1)
      ringbin = 17.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 3 && m_id_i.ring() == 2)
      ringbin = 18.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 4 && m_id_i.ring() == 1)
      ringbin = 19.5;
    else if (m_id_i.endcap() == 1 && m_id_i.station() == 4 && m_id_i.ring() == 2)
      ringbin = 20.5;
    m_parent->m_occupancy->Fill(m_id_i.chamber() + 0.5, ringbin);
  }

  return true;
}

bool CSCPairResidualsConstraint::dphidzFromTrack(const std::vector<TrajectoryMeasurement> &measurements,
                                                 const reco::TransientTrack &track,
                                                 const TrackTransformer *trackTransformer,
                                                 double &dphidz) {
  // make a list of hits on all chambers *other* than the ones associated with this constraint
  std::map<int, int> stations;
  TransientTrackingRecHit::ConstRecHitContainer cscHits;
  for (std::vector<TrajectoryMeasurement>::const_iterator measurement = measurements.begin();
       measurement != measurements.end();
       ++measurement) {
    DetId id = measurement->recHit()->geographicalId();
    if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
      CSCDetId cscid(id.rawId());
      CSCDetId chamberId(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber(), 0);
      if (m_parent->m_combineME11 && cscid.station() == 1 && cscid.ring() == 4)
        chamberId = CSCDetId(cscid.endcap(), 1, 1, cscid.chamber(), 0);

      if (chamberId != m_id_i && chamberId != m_id_j) {
        int station = (cscid.endcap() == 1 ? 1 : -1) * cscid.station();
        if (stations.find(station) == stations.end()) {
          stations[station] = 0;
        }
        stations[station]++;

        cscHits.push_back(measurement->recHit());
      }
    }
  }

  // for the fit to be reliable, it needs to cross multiple stations
  int numStations = 0;
  for (std::map<int, int>::const_iterator station = stations.begin(); station != stations.end(); ++station) {
    if (station->second >= m_parent->m_minHitsPerChamber) {
      numStations++;
    }
  }

  if (numStations >= m_parent->m_minStationsInTrackRefits) {
    // refit the track with these hits
    std::vector<Trajectory> trajectories = trackTransformer->transform(track, cscHits);

    if (!trajectories.empty()) {
      const std::vector<TrajectoryMeasurement> &measurements2 = trajectories.begin()->measurements();

      // find the closest TSOS to the Z plane (on both sides)
      bool found_plus = false;
      bool found_minus = false;
      TrajectoryStateOnSurface tsos_plus, tsos_minus;
      for (std::vector<TrajectoryMeasurement>::const_iterator measurement = measurements2.begin();
           measurement != measurements2.end();
           ++measurement) {
        double z = measurement->recHit()->globalPosition().z();
        if (z > m_Zplane) {
          if (!found_plus || fabs(z - m_Zplane) < fabs(tsos_plus.globalPosition().z() - m_Zplane)) {
            tsos_plus = TrajectoryStateCombiner().combine(measurement->forwardPredictedState(),
                                                          measurement->backwardPredictedState());
          }
          if (tsos_plus.isValid())
            found_plus = true;
        } else {
          if (!found_minus || fabs(z - m_Zplane) < fabs(tsos_minus.globalPosition().z() - m_Zplane)) {
            tsos_minus = TrajectoryStateCombiner().combine(measurement->forwardPredictedState(),
                                                           measurement->backwardPredictedState());
          }
          if (tsos_minus.isValid())
            found_minus = true;
        }
      }

      // propagate from the closest TSOS to the Z plane (from both sides, if possible)
      TrajectoryStateOnSurface from_plus, from_minus;
      if (found_plus) {
        from_plus = m_propagator->propagate(tsos_plus, *m_Zsurface);
      }
      if (found_minus) {
        from_minus = m_propagator->propagate(tsos_minus, *m_Zsurface);
      }

      // if you have two sides, merge them
      TrajectoryStateOnSurface merged;
      if (found_plus && from_plus.isValid() && found_minus && from_minus.isValid()) {
        merged = TrajectoryStateCombiner().combine(from_plus, from_minus);
      } else if (found_plus && from_plus.isValid()) {
        merged = from_plus;
      } else if (found_minus && from_minus.isValid()) {
        merged = from_minus;
      } else
        return false;

      // if, after all that, we have a good fit-and-propagation, report the direction
      if (merged.isValid()) {
        double angle = merged.globalPosition().phi() + M_PI / 2.;
        GlobalVector direction = merged.globalDirection();
        double dxdz = direction.x() / direction.z();
        double dydz = direction.y() / direction.z();
        dphidz = (dxdz * cos(angle) + dydz * sin(angle)) / merged.globalPosition().perp();
        return true;
      }

    }  // end if refit successful
  }    // end if enough hits
  return false;
}

void CSCPairResidualsConstraint::write(std::ofstream &output) {
  output << std::setprecision(14) << std::fixed;
  output << "CSCPairResidualsConstraint " << m_identifier << " " << i() << " " << j() << " " << m_sumN << " " << m_sum1
         << " " << m_sumx << " " << m_sumy << " " << m_sumxx << " " << m_sumyy << " " << m_sumxy << " EOLN"
         << std::endl;
}

void CSCPairResidualsConstraint::read(std::vector<std::ifstream *> &input, std::vector<std::string> &filenames) {
  m_sumN = 0;
  m_sum1 = 0.;
  m_sumx = 0.;
  m_sumy = 0.;
  m_sumxx = 0.;
  m_sumyy = 0.;
  m_sumxy = 0.;

  std::vector<std::ifstream *>::const_iterator inputiter = input.begin();
  std::vector<std::string>::const_iterator filename = filenames.begin();
  for (; inputiter != input.end(); ++inputiter, ++filename) {
    int linenumber = 0;
    bool touched = false;
    while (!(*inputiter)->eof()) {
      linenumber++;
      std::string name, eoln;
      unsigned int identifier;
      int i, j;
      int sumN;
      double sum1, sumx, sumy, sumxx, sumyy, sumxy;

      (**inputiter) >> name >> identifier >> i >> j >> sumN >> sum1 >> sumx >> sumy >> sumxx >> sumyy >> sumxy >> eoln;

      if (!(*inputiter)->eof() && (name != "CSCPairResidualsConstraint" || eoln != "EOLN"))
        throw cms::Exception("CorruptTempFile")
            << "Temporary file " << *filename << " is incorrectly formatted on line " << linenumber << std::endl;

      if (identifier == m_identifier) {
        if (i != m_i || j != m_j)
          throw cms::Exception("CorruptTempFile")
              << "Wrong (i,j) for CSCPairResidualsConstraint " << m_identifier << " (" << m_i << "," << m_j
              << ") in file " << *filename << " on line " << linenumber << std::endl;
        touched = true;

        m_sumN += sumN;
        m_sum1 += sum1;
        m_sumx += sumx;
        m_sumy += sumy;
        m_sumxx += sumxx;
        m_sumyy += sumyy;
        m_sumxy += sumxy;
      }
    }

    (*inputiter)->clear();
    (*inputiter)->seekg(0, std::ios::beg);

    if (!touched)
      throw cms::Exception("CorruptTempFile")
          << "CSCPairResidualsConstraint " << m_identifier << " is missing from file " << *filename << std::endl;
  }
}

void CSCPairResidualsConstraint::calculatePhi(
    const TransientTrackingRecHit *hit, double &phi, double &phierr2, bool doRphi, bool globalPhi) {
  align::LocalPoint pos = hit->localPosition();
  DetId id = hit->geographicalId();
  CSCDetId cscid = CSCDetId(id.rawId());

  double r = 0.;
  if (globalPhi) {
    phi = hit->globalPosition().phi();
    r = hit->globalPosition().perp();

    //     double sinAngle = sin(phi);
    //     double cosAngle = cos(phi);
    //     double xx = hit->globalPositionError().cxx();
    //     double xy = hit->globalPositionError().cyx();
    //     double yy = hit->globalPositionError().cyy();
    //     phierr2 = (xx*cosAngle*cosAngle + 2.*xy*sinAngle*cosAngle + yy*sinAngle*sinAngle) / (r*r);
  } else {
    // these constants are related to the way CSC chambers are built--- really constant!
    const double R_ME11 = 181.5;
    const double R_ME12 = 369.7;
    const double R_ME21 = 242.7;
    const double R_ME31 = 252.7;
    const double R_ME41 = 262.65;
    const double R_MEx2 = 526.5;

    double R = 0.;
    if (cscid.station() == 1 && (cscid.ring() == 1 || cscid.ring() == 4))
      R = R_ME11;
    else if (cscid.station() == 1 && cscid.ring() == 2)
      R = R_ME12;
    else if (cscid.station() == 2 && cscid.ring() == 1)
      R = R_ME21;
    else if (cscid.station() == 3 && cscid.ring() == 1)
      R = R_ME31;
    else if (cscid.station() == 4 && cscid.ring() == 1)
      R = R_ME41;
    else if (cscid.station() > 1 && cscid.ring() == 2)
      R = R_MEx2;
    else
      assert(false);
    r = (pos.y() + R);

    phi = atan2(pos.x(), r);

    if (cscid.endcap() == 1 && cscid.station() >= 3)
      phi *= -1;
    else if (cscid.endcap() == 2 && cscid.station() <= 2)
      phi *= -1;
  }

  int strip = m_cscGeometry->layer(id)->geometry()->nearestStrip(pos);
  double angle = m_cscGeometry->layer(id)->geometry()->stripAngle(strip) - M_PI / 2.;
  double sinAngle = sin(angle);
  double cosAngle = cos(angle);
  double xx = hit->localPositionError().xx();
  double xy = hit->localPositionError().xy();
  double yy = hit->localPositionError().yy();
  phierr2 = (xx * cosAngle * cosAngle + 2. * xy * sinAngle * cosAngle + yy * sinAngle * sinAngle) / (r * r);

  if (doRphi) {
    phi *= r;
    phierr2 *= r * r;
  }
}

bool CSCPairResidualsConstraint::isFiducial(std::vector<const TransientTrackingRecHit *> &hits, bool is_i) {
  // these constants are related to the way CSC chambers are built--- really constant!
  const double cut_ME11 = 0.086;
  const double cut_ME12 = 0.090;
  const double cut_MEx1 = 0.180;
  const double cut_MEx2 = 0.090;

  double sum1 = 0.;
  double sumx = 0.;
  double sumy = 0.;
  double sumxx = 0.;
  double sumxy = 0.;
  for (std::vector<const TransientTrackingRecHit *>::const_iterator hit = hits.begin(); hit != hits.end(); ++hit) {
    double phi, phierr2;
    calculatePhi(*hit, phi, phierr2);
    double z = (is_i ? m_cscGeometry->idToDet(m_id_i)->surface() : m_cscGeometry->idToDet(m_id_j)->surface())
                   .toLocal((*hit)->globalPosition())
                   .z();

    if (m_parent->m_makeHistograms) {
      if (m_id_i.station() == 1 && (m_id_i.ring() == 1 || m_id_i.ring() == 4)) {
        m_parent->m_fiducial_ME11->Fill(fabs(phi), sqrt(phierr2));
      } else if (m_id_i.station() == 1 && m_id_i.ring() == 2) {
        m_parent->m_fiducial_ME12->Fill(fabs(phi), sqrt(phierr2));
      } else if (m_id_i.station() > 1 && m_id_i.ring() == 1) {
        m_parent->m_fiducial_MEx1->Fill(fabs(phi), sqrt(phierr2));
      } else if (m_id_i.station() > 1 && m_id_i.ring() == 2) {
        m_parent->m_fiducial_MEx2->Fill(fabs(phi), sqrt(phierr2));
      }
    }

    double weight = 1.;
    if (m_parent->m_useHitWeights)
      weight = 1. / phierr2;
    sum1 += weight;
    sumx += weight * z;
    sumy += weight * phi;
    sumxx += weight * z * z;
    sumxy += weight * z * phi;
  }
  double delta = (sum1 * sumxx) - (sumx * sumx);
  if (delta == 0.)
    return false;
  double intercept = ((sumxx * sumy) - (sumx * sumxy)) / delta;
  double slope = ((sum1 * sumxy) - (sumx * sumy)) / delta;

  double phi1 = intercept + slope * (is_i ? m_iZ1 : m_jZ1);
  double phi6 = intercept + slope * (is_i ? m_iZ6 : m_jZ6);

  if (m_id_i.station() == 1 && (m_id_i.ring() == 1 || m_id_i.ring() == 4)) {
    return (fabs(phi1) < cut_ME11 && fabs(phi6) < cut_ME11);
  } else if (m_id_i.station() == 1 && m_id_i.ring() == 2) {
    return (fabs(phi1) < cut_ME12 && fabs(phi6) < cut_ME12);
  } else if (m_id_i.station() > 1 && m_id_i.ring() == 1) {
    return (fabs(phi1) < cut_MEx1 && fabs(phi6) < cut_MEx1);
  } else if (m_id_i.station() > 1 && m_id_i.ring() == 2) {
    return (fabs(phi1) < cut_MEx2 && fabs(phi6) < cut_MEx2);
  } else
    assert(false);
}

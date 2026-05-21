#include "DataFormats/Math/interface/deltaPhi.h"

#include "HeavyIonsAnalysis/EGMAnalysis/interface/pfIsoCalculator.h"

void pfIsoCalculator::setVertex(const math::XYZPoint& pv) { vtx_ = pv; }

/*
 meant to do the same function as isInFootprint() in
 https://github.com/CmsHI/cmssw/blob/5c5ccb4013e2ebecfbca0ca433cb14bb36b9f1d3/HeavyIonsAnalysis/PhotonAnalysis/src/pfIsoCalculator.cc#L16
*/
template <class T, class U>
bool pfIsoCalculator::isAssociatedPackedCand(const T& associatedPackedCands, const U& candidate) {
  bool debug = false;
  if (debug) {
    std::cout << "candidate.key() = " << candidate.key() << std::endl;
    std::cout << "associatedPackedCands size = " << associatedPackedCands.size() << std::endl;
  }

  for (auto itr = associatedPackedCands.begin(); itr != associatedPackedCands.end(); ++itr) {
    if (debug) {
      std::cout << "associated Cand key = " << itr->key() << std::endl;
    }

    if (itr->key() == candidate.key()) {
      return true;
    }
  }

  return false;
}

double pfIsoCalculator::getPfIso(
    const pat::Photon& photon, int pfId, double r1, double r2, double threshold, double jWidth, int footprintRemoval) {
  bool debug = false;
  double photonEta = photon.eta();
  double photonPhi = photon.phi();
  double totalEt = 0;

  if (footprintRemoval == pfIsoCalculator::removeSCenergy) {
    totalEt -= (photon.superCluster()->rawEnergy() / std::cosh(photon.superCluster()->eta()));
  }

  for (auto pf = candidatesView->begin(); pf != candidatesView->end(); ++pf) {
    int pfId_ = -999;
    if (usePackedCandidates_) {
      reco::PFCandidate pfTmp;
      pfId_ = pfTmp.translatePdgIdToType(pf->pdgId());
    }

    if (pfId_ != pfId)
      continue;

    double pfEta = pf->eta();
    double pfPhi = pf->phi();

    double dEta = std::abs(photonEta - pfEta);
    double dPhi = reco::deltaPhi(pfPhi, photonPhi);
    double dR2 = dEta * dEta + dPhi * dPhi;
    double pfPt = pf->pt();

    if (footprintRemoval == pfIsoCalculator::removePFcand) {
      // remove the photon itself and associated PF candidates
      /*
      if ( pf->originalObjectRef()->superClusterRef() == photon.superCluster() ) {
        continue;
      }
      */

      edm::Ptr<pat::PackedCandidate> candPtr(candidatesView, pf - candidatesView->begin());
      if (usePackedCandidates_) {
        // std::cout << "numberOfSourceCandidatePtrs = " << photon.numberOfSourceCandidatePtrs() << std::endl;
        if (isAssociatedPackedCand(photon.associatedPackedPFCandidates(), candPtr)) {
          if (debug) {
            std::cout << "found AssociatedPackedPFCand :" << std::endl;
            std::cout << "PF dR = " << std::sqrt(dR2) << std::endl;
            std::cout << "PF dEta = " << dEta << std::endl;
            std::cout << "PF dPhi = " << dPhi << std::endl;
            std::cout << "PF pt = " << pfPt << std::endl;
            std::cout << "pdgId = " << pf->pdgId() << std::endl;
            std::cout << "PF type = " << pfId_ << std::endl;
          }
          continue;
        }
      }
    }

    if (pfId_ == reco::PFCandidate::h) {
      float dz = std::abs(pf->vz() - vtx_.z());
      if (dz > 0.2)
        continue;
      double dxy = ((vtx_.x() - pf->vx()) * pf->py() + (pf->vy() - vtx_.y()) * pf->px()) / pf->pt();
      if (std::abs(dxy) > 0.1)
        continue;
    }

    // Jurassic Cone /////
    if (dR2 > r1 * r1)
      continue;
    if (dR2 < r2 * r2)
      continue;
    if (std::abs(dEta) < jWidth)
      continue;
    if (pfPt < threshold)
      continue;
    totalEt += pfPt;
  }

  return totalEt;
}

double pfIsoCalculator::getPfIsoSubUE(const pat::Photon& photon,
                                      int pfId,
                                      double r1,
                                      double r2,
                                      double threshold,
                                      double jWidth,
                                      int footprintRemoval,
                                      bool excludeCone) {
  double photonEta = photon.eta();
  double photonPhi = photon.phi();
  double totalEt = 0;

  if (footprintRemoval == pfIsoCalculator::removeSCenergy) {
    totalEt -= (photon.superCluster()->rawEnergy() / std::cosh(photon.superCluster()->eta()));
  }

  for (auto pf = candidatesView->begin(); pf != candidatesView->end(); ++pf) {
    int pfId_ = -999;
    if (usePackedCandidates_) {
      reco::PFCandidate pfTmp;
      pfId_ = pfTmp.translatePdgIdToType(pf->pdgId());
    }

    if (pfId_ != pfId)
      continue;

    double pfEta = pf->eta();
    double pfPhi = pf->phi();

    double dEta = std::abs(photonEta - pfEta);
    if (dEta > r1)
      continue;
    if (dEta < jWidth)
      continue;

    if (footprintRemoval == pfIsoCalculator::removePFcand) {
      // remove the photon itself and associated PF candidates
      /*
      if ( pf->originalObjectRef()->superClusterRef() == photon.superCluster() ) {
        continue;
      }
      */

      edm::Ptr<pat::PackedCandidate> candPtr(candidatesView, pf - candidatesView->begin());
      if (usePackedCandidates_) {
        if (isAssociatedPackedCand(photon.associatedPackedPFCandidates(), candPtr)) {
          continue;
        }
      }
    }

    if (pfId_ == reco::PFCandidate::h) {
      float dz = std::abs(pf->vz() - vtx_.z());
      if (dz > 0.2)
        continue;
      double dxy = ((vtx_.x() - pf->vx()) * pf->py() + (pf->vy() - vtx_.y()) * pf->px()) / pf->pt();
      if (std::abs(dxy) > 0.1)
        continue;
    }

    double dPhi = reco::deltaPhi(pfPhi, photonPhi);
    double dR2 = dEta * dEta + dPhi * dPhi;
    double pfPt = pf->pt();

    // Jurassic Cone /////
    if (dR2 < r2 * r2)
      continue;
    if (pfPt < threshold)
      continue;
    totalEt += pfPt;
  }

  double areaStrip = 4 * M_PI * (r1 - jWidth);  // strip area over which UE is estimated
  double areaCone = M_PI * r1 * r1;             // area inside which isolation is to be applied
  if (jWidth > 0) {
    // calculate overlap area between disk with radius r2 and rectangle of width jwidth
    double angTmp = std::acos(jWidth / r1);
    double lenTmp = std::sqrt(r1 * r1 - jWidth * jWidth) * 2;
    double areaTwoTriangles = lenTmp * jWidth;
    double areaTwoArcs = (M_PI - 2 * angTmp) * r1 * r1;
    areaCone -= (areaTwoTriangles + areaTwoArcs);
  }

  double areaInnerCone = 0;
  if (r2 > jWidth) {
    areaInnerCone = M_PI * r2 * r2;
    if (jWidth > 0) {
      double angTmp = std::acos(jWidth / r2);
      double lenTmp = std::sqrt(r2 * r2 - jWidth * jWidth) * 2;
      double areaTwoTriangles = lenTmp * jWidth;
      double areaTwoArcs = (M_PI - 2 * angTmp) * r2 * r2;
      areaInnerCone -= (areaTwoTriangles + areaTwoArcs);
    }
  }
  areaStrip -= areaInnerCone;
  areaCone -= areaInnerCone;

  double coneEt = getPfIso(photon, pfId, r1, r2, threshold, jWidth, footprintRemoval);
  double ueEt = totalEt;
  double ueArea = areaStrip;
  if (excludeCone) {
    ueEt = totalEt - coneEt;
    ueArea = areaStrip - areaCone;
  }

  return coneEt - ueEt * (areaCone / ueArea);
}

double pfIsoCalculator::getPfIso(const reco::GsfElectron& ele, int pfId, double r1, double r2, double threshold) {
  double eleEta = ele.eta();
  double elePhi = ele.phi();
  double totalEt = 0.;

  for (auto pf = candidatesView->begin(); pf != candidatesView->end(); ++pf) {
    int pfId_ = -999;
    if (usePackedCandidates_) {
      reco::PFCandidate pfTmp;
      pfId_ = pfTmp.translatePdgIdToType(pf->pdgId());
    }

    if (pfId_ != pfId)
      continue;

    double pfEta = pf->eta();
    double pfPhi = pf->phi();

    double dEta = std::abs(eleEta - pfEta);
    double dPhi = reco::deltaPhi(pfPhi, elePhi);
    double dR2 = dEta * dEta + dPhi * dPhi;
    double pfPt = pf->pt();

    // remove electron itself
    if (pfId_ == reco::PFCandidate::e)
      continue;

    if (pfId_ == reco::PFCandidate::h) {
      float dz = std::abs(pf->vz() - vtx_.z());
      if (dz > 0.2)
        continue;
      double dxy = ((vtx_.x() - pf->vx()) * pf->py() + (pf->vy() - vtx_.y()) * pf->px()) / pf->pt();
      if (std::abs(dxy) > 0.1)
        continue;
    }

    // inside the cone size
    if (dR2 > r1 * r1)
      continue;
    if (dR2 < r2 * r2)
      continue;
    if (pfPt < threshold)
      continue;
    totalEt += pfPt;
  }

  return totalEt;
}

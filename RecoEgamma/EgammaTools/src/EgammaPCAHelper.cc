
#include "RecoEgamma/EgammaTools/interface/EgammaPCAHelper.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <iostream>
#include <memory>

using namespace hgcal;

EGammaPCAHelper::EGammaPCAHelper()
    :  // Thickness correction to dEdx weights
      // (100um, 200um, 300um silicon)
      // See RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi
      invThicknessCorrection_({1. / 1.132, 1. / 1.092, 1. / 1.084}),
      pca_(new TPrincipal(3, "D")) {
  hitMap_ = nullptr;
  debug_ = false;
}

void EGammaPCAHelper::setHitMap(const std::unordered_map<DetId, const HGCRecHit*>* hitMap) {
  hitMap_ = hitMap;
  pcaIteration_ = 0;
}

void EGammaPCAHelper::setRecHitTools(const hgcal::RecHitTools* recHitTools) {
  recHitTools_ = recHitTools;
  maxlayer_ = recHitTools_->lastLayerBH();
}

void EGammaPCAHelper::storeRecHits(const reco::HGCalMultiCluster& cluster) {
  theCluster_ = &cluster;
  std::vector<std::pair<DetId, float>> result;
  for (reco::HGCalMultiCluster::component_iterator it = cluster.begin(); it != cluster.end(); it++) {
    const std::vector<std::pair<DetId, float>>& hf = (*it)->hitsAndFractions();
    result.insert(result.end(), hf.begin(), hf.end());
  }
  storeRecHits(result);
}

void EGammaPCAHelper::storeRecHits(const reco::CaloCluster& cluster) {
  theCluster_ = &cluster;
  storeRecHits(cluster.hitsAndFractions());
}

void EGammaPCAHelper::storeRecHits(const std::vector<std::pair<DetId, float>>& hf) {
  std::vector<double> pcavars;
  pcavars.resize(3, 0.);
  theSpots_.clear();
  pcaIteration_ = 0;

  sigu_ = 0.;
  sigv_ = 0.;
  sigp_ = 0.;
  sige_ = 0.;

  unsigned hfsize = hf.size();
  if (debug_)
    std::cout << "The seed cluster constains " << hfsize << " hits " << std::endl;

  if (hfsize == 0)
    return;

  for (unsigned int j = 0; j < hfsize; j++) {
    unsigned int layer = recHitTools_->getLayerWithOffset(hf[j].first);

    const DetId rh_detid = hf[j].first;
    std::unordered_map<DetId, const HGCRecHit*>::const_iterator itcheck = hitMap_->find(rh_detid);
    if (itcheck == hitMap_->end()) {
      edm::LogWarning("EgammaPCAHelper") << " Big problem, unable to find a hit " << rh_detid.rawId() << " "
                                         << rh_detid.det() << " " << HGCalDetId(rh_detid) << std::endl;
      continue;
    }
    if (debug_) {
      std::cout << "DetId " << rh_detid.rawId() << " " << layer << " " << itcheck->second->energy() << std::endl;
      std::cout << " Hit " << itcheck->second << " " << itcheck->second->energy() << std::endl;
    }
    float fraction = hf[j].second;

    int thickIndex = recHitTools_->getSiThickIndex(rh_detid);
    double mip = dEdXWeights_[layer] * 0.001;  // convert in GeV
    if (thickIndex > -1 and thickIndex < 3)
      mip *= invThicknessCorrection_[thickIndex];

    pcavars[0] = recHitTools_->getPosition(rh_detid).x();
    pcavars[1] = recHitTools_->getPosition(rh_detid).y();
    pcavars[2] = recHitTools_->getPosition(rh_detid).z();
    if (pcavars[2] == 0.)
      edm::LogWarning("EgammaPCAHelper") << " Problem, hit with z =0 ";
    else {
      Spot mySpot(rh_detid, itcheck->second->energy(), pcavars, layer, fraction, mip);
      theSpots_.push_back(mySpot);
    }
  }
  if (debug_) {
    std::cout << " Stored " << theSpots_.size() << " hits " << std::endl;
  }
}

void EGammaPCAHelper::computePCA(float radius, bool withHalo) {
  // very important - to reset
  pca_ = std::make_unique<TPrincipal>(3, "D");
  bool initialCalculation = radius < 0;
  if (debug_)
    std::cout << " Initial calculation " << initialCalculation << std::endl;
  if (initialCalculation && withHalo) {
    edm::LogWarning("EGammaPCAHelper") << "Warning - in the first iteration, the halo hits are excluded " << std::endl;
    withHalo = false;
  }

  float radius2 = radius * radius;
  if (!initialCalculation) {
    math::XYZVector mainAxis(axis_);
    mainAxis.unit();
    math::XYZVector phiAxis(barycenter_.x(), barycenter_.y(), 0);
    math::XYZVector udir(mainAxis.Cross(phiAxis));
    udir = udir.unit();
    trans_ = Transform3D(Point(barycenter_),
                         Point(barycenter_ + axis_),
                         Point(barycenter_ + udir),
                         Point(0, 0, 0),
                         Point(0., 0., 1.),
                         Point(1., 0., 0.));
  }

  std::set<int> layers;
  for (const auto& spot : theSpots_) {
    if (spot.layer() > recHitTools_->lastLayerEE())
      continue;
    if (!withHalo && (!spot.isCore()))
      continue;
    if (initialCalculation) {
      // initial calculation, take only core hits
      if (!spot.isCore())
        continue;
      layers.insert(spot.layer());
      for (int i = 0; i < spot.multiplicity(); ++i)
        pca_->AddRow(spot.row());
    } else {
      // use a cylinder, include all hits
      math::XYZPoint local = trans_(Point(spot.row()[0], spot.row()[1], spot.row()[2]));
      if (local.Perp2() > radius2)
        continue;
      layers.insert(spot.layer());
      for (int i = 0; i < spot.multiplicity(); ++i)
        pca_->AddRow(spot.row());
    }
  }
  if (debug_)
    std::cout << " Nlayers " << layers.size() << std::endl;
  if (layers.size() < 3) {
    pcaIteration_ = -1;
    return;
  }
  pca_->MakePrincipals();
  ++pcaIteration_;
  const TVectorD& means = *(pca_->GetMeanValues());
  const TMatrixD& eigens = *(pca_->GetEigenVectors());

  barycenter_ = math::XYZPoint(means[0], means[1], means[2]);
  axis_ = math::XYZVector(eigens(0, 0), eigens(1, 0), eigens(2, 0));
  if (axis_.z() * barycenter_.z() < 0.0) {
    axis_ = -1. * axis_;
  }
}

void EGammaPCAHelper::computeShowerWidth(float radius, bool withHalo) {
  sigu_ = 0.;
  sigv_ = 0.;
  sigp_ = 0.;
  sige_ = 0.;
  double cyl_ene = 0.;

  float radius2 = radius * radius;
  for (const auto& spot : theSpots_) {
    Point globalPoint(spot.row()[0], spot.row()[1], spot.row()[2]);
    math::XYZPoint local = trans_(globalPoint);
    if (local.Perp2() > radius2)
      continue;

    // Select halo hits or not
    if (withHalo && spot.fraction() < 0)
      continue;
    if (!withHalo && !(spot.isCore()))
      continue;

    sige_ += (globalPoint.eta() - theCluster_->eta()) * (globalPoint.eta() - theCluster_->eta()) * spot.energy();
    sigp_ += deltaPhi(globalPoint.phi(), theCluster_->phi()) * deltaPhi(globalPoint.phi(), theCluster_->phi()) *
             spot.energy();

    sigu_ += local.x() * local.x() * spot.energy();
    sigv_ += local.y() * local.y() * spot.energy();
    cyl_ene += spot.energy();
  }

  if (cyl_ene > 0.) {
    const double inv_cyl_ene = 1. / cyl_ene;
    sigu_ = sigu_ * inv_cyl_ene;
    sigv_ = sigv_ * inv_cyl_ene;
    sigp_ = sigp_ * inv_cyl_ene;
    sige_ = sige_ * inv_cyl_ene;
  }
  sigu_ = std::sqrt(sigu_);
  sigv_ = std::sqrt(sigv_);
  sigp_ = std::sqrt(sigp_);
  sige_ = std::sqrt(sige_);
}

bool EGammaPCAHelper::checkIteration() const {
  if (pcaIteration_ == 0) {
    if (debug_)
      std::cout << " The PCA has not been run yet " << std::endl;
    return false;
  } else if (pcaIteration_ == 1) {
    if (debug_)
      std::cout << " The PCA has been run only once - careful " << std::endl;
    return false;
  } else if (pcaIteration_ == -1) {
    if (debug_)
      std::cout << " Not enough layers to perform PCA " << std::endl;
    return false;
  }
  return true;
}

void EGammaPCAHelper::clear() {
  theSpots_.clear();
  pcaIteration_ = 0;
  sigu_ = 0.;
  sigv_ = 0.;
  sigp_ = 0.;
  sige_ = 0.;
}

LongDeps EGammaPCAHelper::energyPerLayer(float radius, bool withHalo) {
  if (debug_)
    checkIteration();
  std::set<int> layers;
  float radius2 = radius * radius;
  std::vector<float> energyPerLayer(maxlayer_ + 1, 0.f);
  math::XYZVector mainAxis(axis_);
  mainAxis.unit();
  math::XYZVector phiAxis(barycenter_.x(), barycenter_.y(), 0);
  math::XYZVector udir(mainAxis.Cross(phiAxis));
  udir = udir.unit();
  trans_ = Transform3D(Point(barycenter_),
                       Point(barycenter_ + axis_),
                       Point(barycenter_ + udir),
                       Point(0, 0, 0),
                       Point(0., 0., 1.),
                       Point(1., 0., 0.));
  float energyEE = 0.;
  float energyFH = 0.;
  float energyBH = 0.;

  for (const auto& spot : theSpots_) {
    if (!withHalo && !spot.isCore())
      continue;
    math::XYZPoint local = trans_(Point(spot.row()[0], spot.row()[1], spot.row()[2]));
    if (local.Perp2() > radius2)
      continue;
    energyPerLayer[spot.layer()] += spot.energy();
    layers.insert(spot.layer());
    if (spot.detId().det() == DetId::HGCalEE or spot.subdet() == HGCEE) {
      energyEE += spot.energy();
    } else if (spot.detId().det() == DetId::HGCalHSi or spot.subdet() == HGCHEF) {
      energyFH += spot.energy();
    } else if (spot.detId().det() == DetId::HGCalHSc or spot.subdet() == HGCHEB) {
      energyBH += spot.energy();
    }
  }
  return LongDeps(radius, energyPerLayer, energyEE, energyFH, energyBH, layers);
}

void EGammaPCAHelper::printHits(float radius) const {
  unsigned nSpots = theSpots_.size();
  float radius2 = radius * radius;
  for (unsigned i = 0; i < nSpots; ++i) {
    Spot spot(theSpots_[i]);
    math::XYZPoint local = trans_(Point(spot.row()[0], spot.row()[1], spot.row()[2]));
    if (local.Perp2() < radius2) {
      std::cout << i << "  " << theSpots_[i].detId().rawId() << " " << theSpots_[i].layer() << " "
                << theSpots_[i].energy() << " " << theSpots_[i].isCore();
      std::cout << " " << std::sqrt(local.Perp2()) << std::endl;
    }
  }
}

float EGammaPCAHelper::findZFirstLayer(const LongDeps& ld) const {
  unsigned int firstLayer = 0;
  for (unsigned il = 1; il <= maxlayer_; ++il) {
    if (ld.energyPerLayer()[il] > 0.) {
      firstLayer = il;
      break;
    }
  }
  // Make dummy DetId to get abs(z) for layer
  return recHitTools_->getPositionLayer(firstLayer).z();
}

float EGammaPCAHelper::clusterDepthCompatibility(const LongDeps& ld,
                                                 float& measuredDepth,
                                                 float& expectedDepth,
                                                 float& expectedSigma) {
  expectedDepth = -999.;
  expectedSigma = -999.;
  measuredDepth = -999.;
  if (!checkIteration())
    return -999.;

  float z = findZFirstLayer(ld);
  math::XYZVector dir = axis_.unit();
  measuredDepth = std::abs((z - std::abs(barycenter_.z())) / dir.z());
  return showerDepth_.getClusterDepthCompatibility(measuredDepth, ld.energyEE(), expectedDepth, expectedSigma);
}

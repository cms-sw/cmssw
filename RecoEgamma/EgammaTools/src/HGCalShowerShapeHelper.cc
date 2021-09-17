#include "RecoEgamma/EgammaTools/interface/HGCalShowerShapeHelper.h"

#include <Math/Vector3D.h>
#include <TMatrixD.h>
#include <TVectorD.h>

#include <algorithm>

const double HGCalShowerShapeHelper::kLDWaferCellSize_ = 0.698;
const double HGCalShowerShapeHelper::kHDWaferCellSize_ = 0.465;

HGCalShowerShapeHelper::ShowerShapeCalc::ShowerShapeCalc(
    std::shared_ptr<const hgcal::RecHitTools> recHitTools,
    std::shared_ptr<const std::unordered_map<uint32_t, const reco::PFRecHit *>> pfRecHitPtrMap,
    const std::vector<std::pair<DetId, float>> &hitsAndFracs,
    const double rawEnergy,
    const double minHitE,
    const double minHitET,
    const int minLayer,
    const int maxLayer,
    const DetId::Detector subDet)
    : recHitTools_(recHitTools),
      pfRecHitPtrMap_(pfRecHitPtrMap),
      rawEnergy_(rawEnergy),
      minHitE_(minHitE),
      minHitET_(minHitET),
      minHitET2_(minHitET * minHitET),
      minLayer_(minLayer),
      maxLayer_(maxLayer <= 0 ? recHitTools_->lastLayerEE() : maxLayer),
      nLayer_(maxLayer_ - minLayer_ + 1),
      subDet_(subDet) {
  assert(nLayer_ > 0);
  setFilteredHitsAndFractions(hitsAndFracs);
  setLayerWiseInfo();
}

double HGCalShowerShapeHelper::ShowerShapeCalc::getCellSize(DetId detId) const {
  return recHitTools_->getSiThickIndex(detId) == 0 ? kHDWaferCellSize_ : kLDWaferCellSize_;
}

double HGCalShowerShapeHelper::ShowerShapeCalc::getRvar(double cylinderR, bool useFractions, bool useCellSize) const {
  if (hitsAndFracs_.empty()) {
    return 0.0;
  }

  if (rawEnergy_ <= 0.0) {
    edm::LogWarning("HGCalShowerShapeHelper")
        << "Encountered negative or zero energy for HGCal R-variable denominator: " << rawEnergy_ << std::endl;
  }

  double cylinderR2 = cylinderR * cylinderR;

  double rVar = 0.0;

  auto hitEnergyIter = useFractions ? hitEnergiesWithFracs_.begin() : hitEnergies_.begin();

  hitEnergyIter--;

  for (const auto &hnf : hitsAndFracs_) {
    hitEnergyIter++;

    DetId hitId = hnf.first;

    int hitLayer = recHitTools_->getLayer(hitId) - 1;

    const auto &hitPos = recHitTools_->getPosition(hitId);
    ROOT::Math::XYZVector hitXYZ(hitPos.x(), hitPos.y(), hitPos.z());

    auto distXYZ = hitXYZ - layerCentroids_[hitLayer];

    double r2 = distXYZ.x() * distXYZ.x() + distXYZ.y() * distXYZ.y();

    // Including the cell size seems to make the variable less sensitive to the HD/LD transition region
    if (useCellSize) {
      if (std::sqrt(r2) > cylinderR + getCellSize(hitId)) {
        continue;
      }
    }

    else if (r2 > cylinderR2) {
      continue;
    }

    rVar += *hitEnergyIter;
  }

  rVar /= rawEnergy_;

  return rVar;
}

HGCalShowerShapeHelper::ShowerWidths HGCalShowerShapeHelper::ShowerShapeCalc::getPCAWidths(double cylinderR,
                                                                                           bool useFractions) const {
  if (hitsAndFracs_.empty()) {
    return ShowerWidths();
  }

  double cylinderR2 = cylinderR * cylinderR;

  TMatrixD covMat(3, 3);

  double dxdx = 0.0;
  double dydy = 0.0;
  double dzdz = 0.0;

  double dxdy = 0.0;
  double dydz = 0.0;
  double dzdx = 0.0;

  double totalW = 0.0;

  auto hitEnergyIter = useFractions ? hitEnergiesWithFracs_.begin() : hitEnergies_.begin();

  int nHit = 0;
  hitEnergyIter--;

  for (const auto &hnf : hitsAndFracs_) {
    hitEnergyIter++;

    DetId hitId = hnf.first;

    const auto &hitPos = recHitTools_->getPosition(hitId);
    ROOT::Math::XYZVector hitXYZ(hitPos.x(), hitPos.y(), hitPos.z());

    int hitLayer = recHitTools_->getLayer(hitId) - 1;

    ROOT::Math::XYZVector radXYZ = hitXYZ - layerCentroids_[hitLayer];

    double r2 = radXYZ.x() * radXYZ.x() + radXYZ.y() * radXYZ.y();

    if (r2 > cylinderR2) {
      continue;
    }

    ROOT::Math::XYZVector dXYZ = hitXYZ - centroid_;

    double weight = *hitEnergyIter;
    totalW += weight;

    dxdx += weight * dXYZ.x() * dXYZ.x();
    dydy += weight * dXYZ.y() * dXYZ.y();
    dzdz += weight * dXYZ.z() * dXYZ.z();

    dxdy += weight * dXYZ.x() * dXYZ.y();
    dydz += weight * dXYZ.y() * dXYZ.z();
    dzdx += weight * dXYZ.z() * dXYZ.x();

    nHit++;
  }

  if (!totalW || nHit < 2) {
    return ShowerWidths();
  }

  dxdx /= totalW;
  dydy /= totalW;
  dzdz /= totalW;

  dxdy /= totalW;
  dydz /= totalW;
  dzdx /= totalW;

  covMat(0, 0) = dxdx;
  covMat(1, 1) = dydy;
  covMat(2, 2) = dzdz;

  covMat(0, 1) = covMat(1, 0) = dxdy;
  covMat(0, 2) = covMat(2, 0) = dzdx;
  covMat(1, 2) = covMat(2, 1) = dydz;

  if (!covMat.Sum()) {
    return ShowerWidths();
  }

  // Get eigen values and vectors
  TVectorD eigVals(3);
  TMatrixD eigVecMat(3, 3);

  eigVecMat = covMat.EigenVectors(eigVals);

  ShowerWidths returnWidths;

  returnWidths.sigma2xx = dxdx;
  returnWidths.sigma2yy = dydy;
  returnWidths.sigma2zz = dzdz;

  returnWidths.sigma2xy = dxdy;
  returnWidths.sigma2yz = dydz;
  returnWidths.sigma2zx = dzdx;

  returnWidths.sigma2uu = eigVals(1);
  returnWidths.sigma2vv = eigVals(2);
  returnWidths.sigma2ww = eigVals(0);

  return returnWidths;
}

std::vector<double> HGCalShowerShapeHelper::ShowerShapeCalc::getEnergyHighestHits(unsigned int nrHits,
                                                                                  bool useFractions) const {
  std::vector<double> sortedEnergies(nrHits, 0.);
  const auto &hits = useFractions ? hitEnergiesWithFracs_ : hitEnergies_;
  std::partial_sort_copy(
      hits.begin(), hits.end(), sortedEnergies.begin(), sortedEnergies.end(), std::greater<double>());
  return sortedEnergies;
}

void HGCalShowerShapeHelper::ShowerShapeCalc::setFilteredHitsAndFractions(
    const std::vector<std::pair<DetId, float>> &hitsAndFracs) {
  hitsAndFracs_.clear();
  hitEnergies_.clear();
  hitEnergiesWithFracs_.clear();

  for (const auto &hnf : hitsAndFracs) {
    DetId hitId = hnf.first;
    float hitEfrac = hnf.second;

    int hitLayer = recHitTools_->getLayer(hitId);

    if (hitLayer > nLayer_) {
      continue;
    }

    if (hitId.det() != subDet_) {
      continue;
    }
    auto hitIt = pfRecHitPtrMap_->find(hitId.rawId());
    if (hitIt == pfRecHitPtrMap_->end()) {
      continue;
    }

    const reco::PFRecHit &recHit = *hitIt->second;

    if (recHit.energy() < minHitE_) {
      continue;
    }

    if (recHit.pt2() < minHitET2_) {
      continue;
    }

    // Fill the vectors
    hitsAndFracs_.push_back(hnf);
    hitEnergies_.push_back(recHit.energy());
    hitEnergiesWithFracs_.push_back(recHit.energy() * hitEfrac);
  }
}

void HGCalShowerShapeHelper::ShowerShapeCalc::setLayerWiseInfo() {
  layerEnergies_.clear();
  layerEnergies_.resize(nLayer_);

  layerCentroids_.clear();
  layerCentroids_.resize(nLayer_);

  centroid_.SetXYZ(0, 0, 0);

  int iHit = -1;
  double totalW = 0.0;

  // Compute the centroid per layer
  for (const auto &hnf : hitsAndFracs_) {
    iHit++;

    DetId hitId = hnf.first;

    double weight = hitEnergies_[iHit];
    totalW += weight;

    const auto &hitPos = recHitTools_->getPosition(hitId);
    ROOT::Math::XYZVector hitXYZ(hitPos.x(), hitPos.y(), hitPos.z());

    centroid_ += weight * hitXYZ;

    int hitLayer = recHitTools_->getLayer(hitId) - 1;

    layerEnergies_[hitLayer] += weight;
    layerCentroids_[hitLayer] += weight * hitXYZ;
  }

  int iLayer = -1;

  for (auto &centroid : layerCentroids_) {
    iLayer++;

    if (layerEnergies_[iLayer]) {
      centroid /= layerEnergies_[iLayer];
    }
  }

  if (totalW) {
    centroid_ /= totalW;
  }
}

HGCalShowerShapeHelper::HGCalShowerShapeHelper()
    : recHitTools_(std::make_shared<hgcal::RecHitTools>()),
      pfRecHitPtrMap_(std::make_shared<std::unordered_map<uint32_t, const reco::PFRecHit *>>()) {}

HGCalShowerShapeHelper::HGCalShowerShapeHelper(edm::ConsumesCollector &&sumes)
    : recHitTools_(std::make_shared<hgcal::RecHitTools>()),
      pfRecHitPtrMap_(std::make_shared<std::unordered_map<uint32_t, const reco::PFRecHit *>>()) {
  setTokens(sumes);
}

void HGCalShowerShapeHelper::initPerSetup(const edm::EventSetup &iSetup) {
  recHitTools_->setGeometry(iSetup.getData(caloGeometryToken_));
}

void HGCalShowerShapeHelper::initPerEvent(const std::vector<reco::PFRecHit> &pfRecHits) {
  setPFRecHitPtrMap(pfRecHits);
}

void HGCalShowerShapeHelper::initPerEvent(const edm::EventSetup &iSetup, const std::vector<reco::PFRecHit> &pfRecHits) {
  initPerSetup(iSetup);
  initPerEvent(pfRecHits);
}

HGCalShowerShapeHelper::ShowerShapeCalc HGCalShowerShapeHelper::createCalc(
    const std::vector<std::pair<DetId, float>> &hitsAndFracs,
    double rawEnergy,
    double minHitE,
    double minHitET,
    int minLayer,
    int maxLayer,
    DetId::Detector subDet) const {
  return ShowerShapeCalc(
      recHitTools_, pfRecHitPtrMap_, hitsAndFracs, rawEnergy, minHitE, minHitET, minLayer, maxLayer, subDet);
}

void HGCalShowerShapeHelper::setPFRecHitPtrMap(const std::vector<reco::PFRecHit> &recHits) {
  pfRecHitPtrMap_->clear();

  for (const auto &recHit : recHits) {
    (*pfRecHitPtrMap_)[recHit.detId()] = &recHit;
  }
}

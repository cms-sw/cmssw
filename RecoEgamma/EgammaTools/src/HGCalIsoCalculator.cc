/*
 * HGCalIsoCalculator.cc
 *
 *  Created on: 13 Oct 2017
 *      Author: jkiesele, ncsmith
 */

#include "DataFormats/Math/interface/deltaR.h"
#include "RecoEgamma/EgammaTools/interface/HGCalIsoCalculator.h"

HGCalIsoCalculator::HGCalIsoCalculator()
    : dr2_(0.15 * 0.15), mindr2_(0), rechittools_(nullptr), debug_(false), nlayers_(30) {
  setNRings(5);
}

HGCalIsoCalculator::~HGCalIsoCalculator() {}

void HGCalIsoCalculator::setRecHits(edm::Handle<HGCRecHitCollection> hitsEE,
                                    edm::Handle<HGCRecHitCollection> hitsFH,
                                    edm::Handle<HGCRecHitCollection> hitsBH) {
  recHitsEE_ = hitsEE;
  recHitsFH_ = hitsFH;
  recHitsBH_ = hitsBH;

  if (!rechittools_)
    throw cms::Exception("HGCalIsoCalculator::produceHGCalIso: rechittools not set");

  hitEtaPhiCache_.clear();
  hitEtaPhiCache_.reserve(recHitsEE_->size() + recHitsFH_->size() + recHitsBH_->size());

  // Since HGCal is not projective and the rechits don't cache any
  // eta,phi, we make our own here
  auto makeEtaPhiPair = [this](const auto& hit) {
    const GlobalPoint position = rechittools_->getPosition(hit.id());
    float eta = rechittools_->getEta(position, 0);  //assume vertex at z=0
    float phi = rechittools_->getPhi(position);
    return std::make_pair(eta, phi);
  };

  for (const auto& hit : *recHitsEE_)
    hitEtaPhiCache_.push_back(makeEtaPhiPair(hit));
  for (const auto& hit : *recHitsFH_)
    hitEtaPhiCache_.push_back(makeEtaPhiPair(hit));
  for (const auto& hit : *recHitsBH_)
    hitEtaPhiCache_.push_back(makeEtaPhiPair(hit));
}

void HGCalIsoCalculator::produceHGCalIso(const reco::CaloClusterPtr& seed) {
  if (!rechittools_)
    throw cms::Exception("HGCalIsoCalculator::produceHGCalIso: rechittools not set");

  for (auto& r : isoringdeposits_)
    r = 0;

  // make local temporaries to pass to the lambda
  // avoids recomputing every iteration
  const float seedEta = seed->eta();
  const float seedPhi = seed->phi();
  const std::vector<std::pair<DetId, float>>& seedHitsAndFractions = seed->hitsAndFractions();

  auto checkAndFill = [this, &seedEta, &seedPhi, &seedHitsAndFractions](const HGCRecHit& hit,
                                                                        std::pair<float, float> etaphiVal) {
    float deltar2 = reco::deltaR2(etaphiVal.first, etaphiVal.second, seedEta, seedPhi);
    if (deltar2 > dr2_ || deltar2 < mindr2_)
      return;

    unsigned int layer = rechittools_->getLayerWithOffset(hit.id());
    if (layer >= nlayers_)
      return;

    //do not consider hits associated to the photon cluster
    if (std::none_of(seedHitsAndFractions.begin(), seedHitsAndFractions.end(), [&hit](const auto& seedhit) {
          return hit.id() == seedhit.first;
        })) {
      const unsigned int ring = ringasso_.at(layer);
      isoringdeposits_.at(ring) += hit.energy();
    }
  };

  // The cache order is EE,FH,BH, so we should loop over them the same here
  auto itEtaPhiCache = hitEtaPhiCache_.cbegin();
  for (const auto& hit : *recHitsEE_) {
    checkAndFill(hit, *itEtaPhiCache);
    itEtaPhiCache++;
  }
  for (const auto& hit : *recHitsFH_) {
    checkAndFill(hit, *itEtaPhiCache);
    itEtaPhiCache++;
  }
  for (const auto& hit : *recHitsBH_) {
    checkAndFill(hit, *itEtaPhiCache);
    itEtaPhiCache++;
  }
}

void HGCalIsoCalculator::setNRings(const size_t nrings) {
  if (nrings > nlayers_)
    throw std::logic_error("PhotonHGCalIsoCalculator::setNRings: max number of rings reached");

  ringasso_.clear();
  isoringdeposits_.clear();
  unsigned int separator = nlayers_ / nrings;
  size_t counter = 0;
  for (size_t i = 0; i < nlayers_ + 1; i++) {
    ringasso_.push_back(counter);
    //the last ring might be larger.
    if (i && !(i % separator) && (int)counter < (int)nrings - 1) {
      counter++;
    }
  }
  isoringdeposits_.resize(nrings, 0);
}

const float HGCalIsoCalculator::getIso(const unsigned int ring) const {
  if (ring >= isoringdeposits_.size())
    throw cms::Exception("HGCalIsoCalculator::getIso: ring index out of range");
  return isoringdeposits_[ring];
}

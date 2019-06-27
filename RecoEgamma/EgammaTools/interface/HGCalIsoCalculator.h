/*
 * PhotonIsoProducer.h
 *
 *  Created on: 13 Oct 2017
 *      Author: jkiesele, ncsmith
 */

#ifndef RecoEgamma_EgammaTools_HGCalIsoCalculator_h
#define RecoEgamma_EgammaTools_HGCalIsoCalculator_h

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

/*
 *
 * This class calculates the energy around the photon/electron in DR=0.15 that is
 * not associated to the seed cluster.
 * The energy is summed in 5 rings (default) between HGCal layer 1 and 30
 * This gives back 5 calorimeter isolation values.
 * Only the first values should be significantly affected by pileup.
 *
 * Usage:
 *
 * PhotonHGCalIsoCalculator prod;
 * prod.setRecHitTools(rechittools)
 * prod.setRecHits(recHitsEEHandle,recHitsFHHandle,recHitsBHHandle)
 *
 * <optional>
 * prod.setDeltaR(0.15)
 * <optional>
 * prod.setNRings(5)
 * <optional>
 * prod.setMinDeltaR(0)
 *
 * for p in photons
 *   prod.produceHGCalIso(p.superCluster()->seed())
 *   a=prod.getIso(0)
 *   b=prod.getIso(1)
 *   c=prod.getIso(2)
 *   d=prod.getIso(3)
 *   e=prod.getIso(4)
 *
 *
 */
class HGCalIsoCalculator {
public:
  HGCalIsoCalculator();

  ~HGCalIsoCalculator();

  void setDeltaR(const float dr) { dr2_ = dr * dr; }

  void setMinDeltaR(const float dr) { mindr2_ = dr * dr; }

  void setRecHitTools(const hgcal::RecHitTools* recHitTools) { rechittools_ = recHitTools; }

  void setNRings(const size_t nrings);

  void setNLayers(unsigned int nLayers) { nlayers_ = nLayers; }

  /// fill - once per event
  void setRecHits(edm::Handle<HGCRecHitCollection> hitsEE,
                  edm::Handle<HGCRecHitCollection> hitsFH,
                  edm::Handle<HGCRecHitCollection> hitsBH);

  void produceHGCalIso(const reco::CaloClusterPtr& seedCluster);

  const float getIso(const unsigned int ring) const;

private:
  std::vector<float> isoringdeposits_;
  std::vector<unsigned int> ringasso_;

  float dr2_, mindr2_;

  const hgcal::RecHitTools* rechittools_;
  edm::Handle<HGCRecHitCollection> recHitsEE_;
  edm::Handle<HGCRecHitCollection> recHitsFH_;
  edm::Handle<HGCRecHitCollection> recHitsBH_;
  std::vector<std::pair<float, float>> hitEtaPhiCache_;
  bool debug_;
  unsigned int nlayers_;
};

#endif /* RecoEgamma_EgammaTools_HGCalIsoCalculator_h */

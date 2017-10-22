/*
 * PhotonIsoProducer.h
 *
 *  Created on: 13 Oct 2017
 *      Author: jkiesele
 */

#ifndef EGAMMATOOL_EGAMMAANALYSIS_HGCALISOPRODUCER_H_
#define EGAMMATOOL_EGAMMAANALYSIS_HGCALISOPRODUCER_H_

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
 * PhotonHGCalIsoProducer prod;
 * prod.setRecHitTools(rechittools)
 * prod.fillHitMap(recHitsEE,recHitsFH,recHitsBH)
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
class HGCalIsoProducer{
public:
    HGCalIsoProducer();

    ~HGCalIsoProducer();

    void setDeltaR(const float& dr){dr2_=dr*dr;}

    void setMinDeltaR(const float& dr){mindr2_=dr*dr;}

    void setRecHitTools(const hgcal::RecHitTools * recHitTools){rechittools_ = recHitTools;}

    void setNRings(const size_t nrings);

    void setNLayers(size_t nLayers){nlayers_=nLayers;}

    /// fill - once per event
    void fillHitMap(const HGCRecHitCollection & rechitsEE,
            const HGCRecHitCollection & rechitsFH,
            const HGCRecHitCollection & rechitsBH);

    /// to set from outside - once per event
    void setHitMap(std::map<DetId, const HGCRecHit *> * hitmap);

    void produceHGCalIso(const reco::CaloClusterPtr & seedCluster);

    const float& getIso(const size_t& ring)const;

private:
    std::vector<float> isoringdeposits_;
    std::vector<size_t> ringasso_;

    float dr2_,mindr2_;

    const hgcal::RecHitTools* rechittools_;
    std::map<DetId, const HGCRecHit *> * allHitMap_;
    bool mapassigned_;
    bool debug_;
    size_t nlayers_;
};


#endif /* EGAMMATOOL_EGAMMAANALYSIS_HGCALISOPRODUCER_H_ */

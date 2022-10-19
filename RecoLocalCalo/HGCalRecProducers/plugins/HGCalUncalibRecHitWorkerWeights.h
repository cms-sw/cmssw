#ifndef RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitRecWorkerWeights_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitRecWorkerWeights_hh

/** \class HGCalUncalibRecHitRecWeightsAlgo
 *
  *  Template used to produce fast-track HGCAL Reco, weight=1
  *
  *
  *  \author Valeri Andreev
  */

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalUncalibRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalUncalibRecHitRecWeightsAlgo.h"
#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"
#include "DataFormats/HGCDigi/interface/HGCSample.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}  // namespace edm

class HGCalUncalibRecHitWorkerWeights : public HGCalUncalibRecHitWorkerBaseClass {
public:
  HGCalUncalibRecHitWorkerWeights(const edm::ParameterSet&, edm::ConsumesCollector iC);
  ~HGCalUncalibRecHitWorkerWeights() override{};

  bool runHGCEE(const edm::ESHandle<HGCalGeometry>& geom,
                const HGCalDigiCollection& digis,
                HGCeeUncalibratedRecHitCollection& result) override;
  bool runHGCHEsil(const edm::ESHandle<HGCalGeometry>& geom,
                   const HGCalDigiCollection& digis,
                   HGChefUncalibratedRecHitCollection& result) override;
  bool runHGCHEscint(const edm::ESHandle<HGCalGeometry>& geom,
                     const HGCalDigiCollection& digis,
                     HGChebUncalibratedRecHitCollection& result) override;
  bool runHGCHFNose(const edm::ESHandle<HGCalGeometry>& geom,
                    const HGCalDigiCollection& digis,
                    HGChfnoseUncalibratedRecHitCollection& result) override;

protected:
  HGCalUncalibRecHitRecWeightsAlgo<HGCalDataFrame> uncalibMaker_ee_;
  HGCalUncalibRecHitRecWeightsAlgo<HGCalDataFrame> uncalibMaker_hef_;
  HGCalUncalibRecHitRecWeightsAlgo<HGCalDataFrame> uncalibMaker_heb_;
  HGCalUncalibRecHitRecWeightsAlgo<HGCalDataFrame> uncalibMaker_hfnose_;

  bool run(const edm::ESHandle<HGCalGeometry>& geom,
           const HGCalDigiCollection& digis,
           HGCalUncalibRecHitRecWeightsAlgo<HGCalDataFrame>& uncalibMaker,
           edm::SortedCollection<HGCUncalibratedRecHit>& result);
};

#endif

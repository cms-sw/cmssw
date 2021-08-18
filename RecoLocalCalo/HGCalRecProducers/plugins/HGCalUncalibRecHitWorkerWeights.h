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

  void set(const edm::EventSetup& es) override;
  bool runHGCEE(const HGCalDigiCollection::const_iterator& digi, HGCeeUncalibratedRecHitCollection& result) override;
  bool runHGCHEsil(const HGCalDigiCollection::const_iterator& digi,
                   HGChefUncalibratedRecHitCollection& result) override;
  bool runHGCHEscint(const HGCalDigiCollection::const_iterator& digi,
                     HGChebUncalibratedRecHitCollection& result) override;
  bool runHGCHFNose(const HGCalDigiCollection::const_iterator& digi,
                    HGChfnoseUncalibratedRecHitCollection& result) override;

protected:
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> ee_geometry_token_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> hef_geometry_token_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> hfnose_geometry_token_;
  HGCalUncalibRecHitRecWeightsAlgo<HGCalDataFrame> uncalibMaker_ee_;
  HGCalUncalibRecHitRecWeightsAlgo<HGCalDataFrame> uncalibMaker_hef_;
  HGCalUncalibRecHitRecWeightsAlgo<HGCalDataFrame> uncalibMaker_heb_;
  HGCalUncalibRecHitRecWeightsAlgo<HGCalDataFrame> uncalibMaker_hfnose_;
};

#endif

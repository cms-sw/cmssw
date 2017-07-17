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
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/HGCDigi/interface/HGCSample.h"
#include "FWCore/Framework/interface/ESHandle.h"


namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}

class HGCalUncalibRecHitWorkerWeights : public HGCalUncalibRecHitWorkerBaseClass {
  
 public:
  HGCalUncalibRecHitWorkerWeights(const edm::ParameterSet&);
  virtual ~HGCalUncalibRecHitWorkerWeights() {};
  
  void set(const edm::EventSetup& es) override;
  bool run1(const edm::Event& evt, const HGCEEDigiCollection::const_iterator & digi, HGCeeUncalibratedRecHitCollection & result) override;
  bool run2(const edm::Event& evt, const HGCHEDigiCollection::const_iterator & digi, HGChefUncalibratedRecHitCollection & result) override;
  bool run3(const edm::Event& evt, const HGCBHDigiCollection::const_iterator & digi, HGChebUncalibratedRecHitCollection & result) override;

 protected:
    
  HGCalUncalibRecHitRecWeightsAlgo<HGCEEDataFrame> uncalibMaker_ee_;
  HGCalUncalibRecHitRecWeightsAlgo<HGCHEDataFrame> uncalibMaker_hef_;
  HGCalUncalibRecHitRecWeightsAlgo<HGCBHDataFrame> uncalibMaker_heb_;

};

#endif

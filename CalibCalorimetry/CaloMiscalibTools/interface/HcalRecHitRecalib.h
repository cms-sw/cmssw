#ifndef _HCALRECHITRECALIB_H
#define _HCALRECHITRECALIB_H

// -*- C++ -*-
//
// Package:    HcalRecHitRecalib
// Class:      HcalRecHitRecalib
//
/**\class HcalRecHitRecalib HcalRecHitRecalib.cc CalibCalorimetry/CaloRecalibTools.src/HcalRecHitRecalib.cc

 Description: Producer to miscalibrate (calibrated) Hcal RecHit 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Luca Malgeri
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapHcal.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalRecHitRecalib : public edm::stream::EDProducer<> {
public:
  explicit HcalRecHitRecalib(const edm::ParameterSet &);
  ~HcalRecHitRecalib() override;

  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  const edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  const edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  const edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topologyToken_;
  const std::string recalibHBHEHits_;
  const std::string recalibHFHits_;
  const std::string recalibHOHits_;

  std::string hcalfile_;
  const std::string hcalfileinpath_;

  CaloMiscalibMapHcal mapHcal_;
  const double refactor_;
  const double refactor_mean_;
};
#endif

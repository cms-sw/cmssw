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
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapHcal.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class HcalRecHitRecalib : public edm::EDProducer
{
public:
    explicit HcalRecHitRecalib(const edm::ParameterSet&);
    ~HcalRecHitRecalib() override;

    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event &, const edm::EventSetup&) override;

private:
    edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
    edm::EDGetTokenT<HORecHitCollection> tok_ho_;
    edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
    std::string RecalibHBHEHits_;
    std::string RecalibHFHits_;
    std::string RecalibHOHits_;

    std::string hcalfile_;
    std::string hcalfileinpath_;

    CaloMiscalibMapHcal mapHcal_;
    double refactor_;
    double refactor_mean_;
};
#endif

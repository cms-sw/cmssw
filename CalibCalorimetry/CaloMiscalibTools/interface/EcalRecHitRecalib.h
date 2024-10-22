#ifndef _ECALRECHITRECALIB_H
#define _ECALRECHITRECALIB_H

// -*- C++ -*-
//
// Package:    EcalRecHitRecalib
// Class:      EcalRecHitRecalib
//
/**\class EcalRecHitRecalib EcalRecHitRecalib.cc CalibCalorimetry/CaloMiscalibTools.src/EcalRecHitRecalib.cc

 Description: Producer to miscalibrate (calibrated) Ecal RecHit 

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
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
//
// class decleration
//

class EcalRecHitRecalib : public edm::stream::EDProducer<> {
public:
  explicit EcalRecHitRecalib(const edm::ParameterSet &);
  ~EcalRecHitRecalib() override;

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  // ----------member data ---------------------------

  const std::string ecalHitsProducer_;
  const std::string barrelHits_;
  const std::string endcapHits_;
  const std::string recalibBarrelHits_;
  const std::string recalibEndcapHits_;
  const double refactor_;
  const double refactor_mean_;

  const edm::EDGetTokenT<EBRecHitCollection> ebRecHitToken_;
  const edm::EDGetTokenT<EERecHitCollection> eeRecHitToken_;
  const edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> intercalibConstsToken_;
  const edm::EDPutTokenT<EBRecHitCollection> barrelHitsToken_;
  const edm::EDPutTokenT<EERecHitCollection> endcapHitsToken_;
};

#endif

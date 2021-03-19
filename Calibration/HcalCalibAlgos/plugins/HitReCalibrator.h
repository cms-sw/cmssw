#ifndef HitReCalibrator_h
#define HitReCalibrator_h

// -*- C++ -*-

// system include files
#include <memory>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"

//
// class declaration
//
namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

namespace cms {

  class HitReCalibrator : public edm::EDProducer {
  public:
    explicit HitReCalibrator(const edm::ParameterSet &);
    ~HitReCalibrator() override;

    void beginJob() override;

    void produce(edm::Event &, const edm::EventSetup &) override;

  private:
    // ----------member data ---------------------------

    bool allowMissingInputs_;

    edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
    edm::EDGetTokenT<HORecHitCollection> tok_ho_;
    edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

    edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_resp_;
  };
}  // end namespace cms
#endif

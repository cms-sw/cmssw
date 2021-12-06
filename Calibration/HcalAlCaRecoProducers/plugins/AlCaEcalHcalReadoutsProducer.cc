// -*- C++ -*-

// system include files
#include <memory>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//
// class decleration
//

class AlCaEcalHcalReadoutsProducer : public edm::global::EDProducer<> {
public:
  explicit AlCaEcalHcalReadoutsProducer(const edm::ParameterSet&);
  ~AlCaEcalHcalReadoutsProducer() override = default;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------member data ---------------------------

  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

  edm::EDPutTokenT<HBHERecHitCollection> put_hbhe_;
  edm::EDPutTokenT<HORecHitCollection> put_ho_;
  edm::EDPutTokenT<HFRecHitCollection> put_hf_;
};

AlCaEcalHcalReadoutsProducer::AlCaEcalHcalReadoutsProducer(const edm::ParameterSet& iConfig) {
  tok_ho_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"));
  tok_hf_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInput"));
  tok_hbhe_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"));

  //register your products
  put_hbhe_ = produces<HBHERecHitCollection>("HBHERecHitCollection");
  put_ho_ = produces<HORecHitCollection>("HORecHitCollection");
  put_hf_ = produces<HFRecHitCollection>("HFRecHitCollection");
}

// ------------ method called to produce the data  ------------
void AlCaEcalHcalReadoutsProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<HBHERecHitCollection> hbhe = iEvent.getHandle(tok_hbhe_);
  if (!hbhe.isValid()) {
    edm::LogVerbatim("AlCaEcalHcal") << "AlCaEcalHcalReadoutProducer: Error! can't get hbhe product!";
    return;
  }

  edm::Handle<HORecHitCollection> ho = iEvent.getHandle(tok_ho_);
  if (!ho.isValid()) {
    edm::LogVerbatim("AlCaEcalHcal") << "AlCaEcalHcalReadoutProducer: Error! can't get ho product!";
  }

  edm::Handle<HFRecHitCollection> hf = iEvent.getHandle(tok_hf_);
  if (!hf.isValid()) {
    edm::LogVerbatim("AlCaEcalHcal") << "AlCaEcalHcalReadoutProducer: Error! can't get hf product!";
  }

  //Put selected information in the event
  iEvent.emplace(put_hbhe_, *hbhe);
  iEvent.emplace(put_ho_, *ho);
  iEvent.emplace(put_hf_, *hf);
}

void AlCaEcalHcalReadoutsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hbheInput", edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("hfInput", edm::InputTag("hfreco"));
  desc.add<edm::InputTag>("hoInput", edm::InputTag("horeco"));
  descriptions.add("alcaEcalHcalReadoutsProducer", desc);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaEcalHcalReadoutsProducer);

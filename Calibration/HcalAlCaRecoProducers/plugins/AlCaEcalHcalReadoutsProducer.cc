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

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

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
  using namespace edm;
  using namespace std;

  edm::Handle<HBHERecHitCollection> hbhe;
  edm::Handle<HORecHitCollection> ho;
  edm::Handle<HFRecHitCollection> hf;

  iEvent.getByToken(tok_hbhe_, hbhe);
  if (!hbhe.isValid()) {
    LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get hbhe product!" << std::endl;
    return;
  }

  iEvent.getByToken(tok_ho_, ho);
  if (!ho.isValid()) {
    LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get ho product!" << std::endl;
  }

  iEvent.getByToken(tok_hf_, hf);
  if (!hf.isValid()) {
    LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get hf product!" << std::endl;
  }

  //Put selected information in the event
  iEvent.emplace(put_hbhe_, *hbhe);
  iEvent.emplace(put_ho_, *ho);
  iEvent.emplace(put_hf_, *hf);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaEcalHcalReadoutsProducer);

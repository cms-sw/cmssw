// -*- C++ -*-
//
// Package:    L1GctInternJetProducer
// Class:      L1GctInternJetProducer
//
/**\class L1GctInternJetProducer \file L1GctInternJetProducer.cc EventFilter/GctRawToDigi/plugins/L1GctInternJetProducer.cc
*/
//
// Original Author:  Alex Tapper
//
//

// system include files
#include <memory>

// user include files
#include "EventFilter/GctRawToDigi/plugins/L1GctInternJetProducer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

L1GctInternJetProducer::L1GctInternJetProducer(const edm::ParameterSet& iConfig)
    : internalJetSource_(iConfig.getParameter<edm::InputTag>("internalJetSource")),
      caloGeomToken_(esConsumes<L1CaloGeometry, L1CaloGeometryRecord>()),
      jetScaleToken_(esConsumes<L1CaloEtScale, L1JetEtScaleRcd>()),
      centralBxOnly_(iConfig.getParameter<bool>("centralBxOnly")) {
  using namespace l1extra;

  produces<L1JetParticleCollection>("Internal");
}

void L1GctInternJetProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  //std::cout << "ARGGHHH!" << std::endl;
  using namespace edm;
  using namespace l1extra;
  using namespace std;

  unique_ptr<L1JetParticleCollection> internJetColl(new L1JetParticleCollection);

  ESHandle<L1CaloGeometry> caloGeomESH = iSetup.getHandle(caloGeomToken_);
  const L1CaloGeometry* caloGeom = &(*caloGeomESH);

  ESHandle<L1CaloEtScale> jetScale = iSetup.getHandle(jetScaleToken_);

  double etSumLSB = jetScale->linearLsb();
  //std::cout << "Inside the Jet producer " << etSumLSB << std::endl;

  Handle<L1GctInternJetDataCollection> hwIntJetCands;
  iEvent.getByLabel(internalJetSource_, hwIntJetCands);
  //std::cout << "At leas Something is happening" <<std::endl;
  if (!hwIntJetCands.isValid()) {
    std::cout << "These aren't the Jets you're looking for" << std::endl;

    LogDebug("L1GctInternJetProducer") << "\nWarning: L1GctJetCandCollection with " << internalJetSource_
                                       << "\nrequested in configuration, but not found in the event." << std::endl;

  } else {
    //std::cout << "apparently the collection was found" <<std::endl;
    L1GctInternJetDataCollection::const_iterator jetItr = hwIntJetCands->begin();
    L1GctInternJetDataCollection::const_iterator jetEnd = hwIntJetCands->end();
    int i;
    for (i = 0; jetItr != jetEnd; ++jetItr, ++i) {
      //std::cout << " JetS a plenty" <<std::endl;
      if (!jetItr->empty() && (!centralBxOnly_ || jetItr->bx() == 0)) {
        double et = (jetItr->oflow() ? (double)0xfff : (double)jetItr->et()) * etSumLSB + 1.e-6;

        //double et = 10.;
        //std::cout << "jetET: " << jetItr->et() <<std::endl;
        //std::cout << "ET was: " << et << std::endl;
        double eta = caloGeom->etaBinCenter(jetItr->regionId());
        double phi = caloGeom->emJetPhiBinCenter(jetItr->regionId());

        internJetColl->push_back(
            L1JetParticle(math::PtEtaPhiMLorentzVector(et, eta, phi, 0.), Ref<L1GctJetCandCollection>(), jetItr->bx()));
      }
    }
  }

  OrphanHandle<L1JetParticleCollection> internalJetHandle = iEvent.put(std::move(internJetColl), "Internal");
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1GctInternJetProducer);

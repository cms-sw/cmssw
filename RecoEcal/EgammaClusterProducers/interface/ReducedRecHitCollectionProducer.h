#ifndef _REDUCEDRECHITPRODUCER_H
#define _REDUCEDRECHITPRODUCER_H

// -*- C++ -*-
//
// Package:    ReducedRecHitCollectionProducer
// Class:      ReducedRecHitCollectionProducer
// 
/**\class ReducedRecHitCollectionProducer ReducedRecHitCollectionProducer.cc Calibration/EcalAlCaRecoProducers/src/ReducedRecHitCollectionProducer.cc

Original author: Paolo Meridiani PH/CMG
 
Implementation:
 <Notes on implementation>
*/



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

class CaloTopology;

class ReducedRecHitCollectionProducer : public edm::stream::EDProducer<> {
   public:
      //! ctor
      explicit ReducedRecHitCollectionProducer(const edm::ParameterSet&);
      ~ReducedRecHitCollectionProducer();
      //! producer
      virtual void produce(edm::Event &, const edm::EventSetup&);

   private:
      // ----------member data ---------------------------
      edm::EDGetTokenT<EcalRecHitCollection>     recHitsToken_;
      std::vector<edm::EDGetTokenT<DetIdCollection> > interestingDetIdCollections_;
      std::string reducedHitsCollection_;
  
};

#endif

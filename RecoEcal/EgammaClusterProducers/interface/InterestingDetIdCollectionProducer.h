#ifndef _INTERESTINGDETIDCOLLECTIONPRODUCER_H
#define _INTERESTINGDETIDCOLLECTIONPRODUCER_H

// -*- C++ -*-
//
// Package:    InterestingDetIdCollectionProducer
// Class:      InterestingDetIdCollectionProducer
// 
/**\class InterestingDetIdCollectionProducer 
Original author: Paolo Meridiani PH/CMG
 
Implementation:
 <Notes on implementation>
*/



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class CaloTopology;

class InterestingDetIdCollectionProducer : public edm::EDProducer {
   public:
      //! ctor
      explicit InterestingDetIdCollectionProducer(const edm::ParameterSet&);
      ~InterestingDetIdCollectionProducer();
      void beginJob (const edm::EventSetup&);
      //! producer
      virtual void produce(edm::Event &, const edm::EventSetup&);

   private:
      // ----------member data ---------------------------
      edm::InputTag recHitsLabel_;
      edm::InputTag basicClusters_;
      std::string interestingDetIdCollection_;
      int minimalEtaSize_;
      int minimalPhiSize_;
      const CaloTopology* caloTopology_;

};

#endif

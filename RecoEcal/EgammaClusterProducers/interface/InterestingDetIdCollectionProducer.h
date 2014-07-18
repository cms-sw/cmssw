#ifndef _INTERESTINGDETIDCOLLECTIONPRODUCER_H
#define _INTERESTINGDETIDCOLLECTIONPRODUCER_H

// -*- C++ -*-
//
// Package:    InterestingDetIdCollectionProducer
// Class:      InterestingDetIdCollectionProducer
// 
/**\class InterestingDetIdCollectionProducer 
Original author: Paolo Meridiani PH/CMG
 
Make a collection of detids to be kept tipically in a AOD rechit collection

The following classes of "interesting id" are considered

    1.in a region around  the seed of the cluster collection specified
      by paramter basicClusters. The size of the region is specified by
      minimalEtaSize_, minimalPhiSize_
 
    2. if the severity of the hit is >= severityLevel_
       If severityLevel=0 this class is ignored

    3. Channels next to dead ones,  keepNextToDead_ is true
    4. Channels next to the EB/EE transition if keepNextToBoundary_ is true
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
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

class CaloTopology;
class EcalSeverityLevelAlgo;

class InterestingDetIdCollectionProducer : public edm::stream::EDProducer<> {
   public:
      //! ctor
      explicit InterestingDetIdCollectionProducer(const edm::ParameterSet&);
      virtual void beginRun (edm::Run const&, const edm::EventSetup&) override final;
      //! producer
      virtual void produce(edm::Event &, const edm::EventSetup&);

   private:
      // ----------member data ---------------------------
      edm::EDGetTokenT<EcalRecHitCollection>         recHitsToken_;
      edm::EDGetTokenT<reco::BasicClusterCollection> basicClustersToken_;
      std::string interestingDetIdCollection_;
      int minimalEtaSize_;
      int minimalPhiSize_;
      const CaloTopology* caloTopology_;

      int severityLevel_;
      const EcalSeverityLevelAlgo * severity_;
      bool  keepNextToDead_;
      bool  keepNextToBoundary_;

};

#endif

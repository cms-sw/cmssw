#ifndef _INTERESTINGDETIDFROMSUPERCLUSTERPRODUCER_H
#define _INTERESTINGDETIDFROMSUPERCLUSTERPRODUCER_H

// -*- C++ -*-
//
// Package:    InterestingDetIdFromSuperClusterProducer
// Class:      InterestingDetIdFromSuperClusterProducer
// 
/**\class InterestingDetIdFromSuperClusterProducer 
Adapted from InterestingDetIdCollectionProducer by J.Bendavid
 
Make a collection of detids to be kept tipically in a AOD rechit collection

The following classes of "interesting id" are considered

    1.All rechits included in all subclusters, plus in a region around  the seed of each subcluster
      The size of the region is specified by
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
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

class CaloTopology;
class EcalSeverityLevelAlgo;

class InterestingDetIdFromSuperClusterProducer : public edm::stream::EDProducer<> {
   public:
      //! ctor
      explicit InterestingDetIdFromSuperClusterProducer(const edm::ParameterSet&);
      virtual void beginRun (edm::Run const&, const edm::EventSetup&) override final;
      //! producer
      virtual void produce(edm::Event &, const edm::EventSetup&);

   private:
      // ----------member data ---------------------------
      edm::EDGetTokenT<EcalRecHitCollection>         recHitsToken_;
      edm::EDGetTokenT<reco::SuperClusterCollection> superClustersToken_;
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

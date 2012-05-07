#ifndef _ELEISODETIDCOLLECTIONPRODUCER_H
#define _ELEISODETIDCOLLECTIONPRODUCER_H

// -*- C++ -*-
//
// Package:    EleIsoDetIdCollectionProducer
// Class:      EleIsoDetIdCollectionProducer
// 
/**\class EleIsoDetIdCollectionProducer 
Original author: Matthew LeBourgeois PH/CMG
Modified from :
RecoEcal/EgammaClusterProducers/{src,interface}/InterestingDetIdCollectionProducer.{h,cc}
by Paolo Meridiani PH/CMG
 
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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

class CaloTopology;

class EleIsoDetIdCollectionProducer : public edm::EDProducer {
   public:
      //! ctor
      explicit EleIsoDetIdCollectionProducer(const edm::ParameterSet&);
      ~EleIsoDetIdCollectionProducer();
      void beginJob ();
      //! producer
      virtual void produce(edm::Event &, const edm::EventSetup&);

   private:
      // ----------member data ---------------------------
      edm::InputTag recHitsLabel_;
      edm::InputTag emObjectLabel_;
      double energyCut_;
      double etCut_;
      double etCandCut_;
      double outerRadius_;
      double innerRadius_;
      std::string interestingDetIdCollection_;
	  int   severityLevelCut_;
      float severityRecHitThreshold_;
      std::string spIdString_;
      float spIdThreshold_;
      EcalSeverityLevelAlgo::SpikeId spId_;
      std::vector<int> v_chstatus_;

};

#endif

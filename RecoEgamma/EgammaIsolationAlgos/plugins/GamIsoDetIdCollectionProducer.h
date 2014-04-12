#ifndef _GAMISODETIDCOLLECTIONPRODUCER_H
#define _GAMISODETIDCOLLECTIONPRODUCER_H

// -*- C++ -*-
//
// Package:    GamIsoDetIdCollectionProducer
// Class:      GamIsoDetIdCollectionProducer
// 
/**\class GamIsoDetIdCollectionProducer 
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

class GamIsoDetIdCollectionProducer : public edm::EDProducer {
   public:
      //! ctor
      explicit GamIsoDetIdCollectionProducer(const edm::ParameterSet&);
      ~GamIsoDetIdCollectionProducer();
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

      std::vector<int> severitiesexclEB_;
      std::vector<int> severitiesexclEE_;
      std::vector<int> flagsexclEB_;
      std::vector<int> flagsexclEE_;
};

#endif



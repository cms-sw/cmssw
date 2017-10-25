#ifndef  _ECALRECHITRECALIB_H
#define  _ECALRECHITRECALIB_H

// -*- C++ -*-
//
// Package:    EcalRecHitRecalib
// Class:      EcalRecHitRecalib
// 
/**\class EcalRecHitRecalib EcalRecHitRecalib.cc CalibCalorimetry/CaloMiscalibTools.src/EcalRecHitRecalib.cc

 Description: Producer to miscalibrate (calibrated) Ecal RecHit 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Luca Malgeri
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class EcalRecHitRecalib : public edm::EDProducer {
   public:
      explicit EcalRecHitRecalib(const edm::ParameterSet&);
      ~EcalRecHitRecalib() override;


      void produce(edm::Event &, const edm::EventSetup&) override;
   private:
      // ----------member data ---------------------------

 std::string ecalHitsProducer_;
 std::string barrelHits_;
 std::string endcapHits_;
 std::string RecalibBarrelHits_;
 std::string RecalibEndcapHits_;
 double refactor_;
 double refactor_mean_;

};

#endif

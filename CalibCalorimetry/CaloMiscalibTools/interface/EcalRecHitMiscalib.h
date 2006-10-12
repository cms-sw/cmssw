// -*- C++ -*-
//
// Package:    EcalRecHitMiscalib
// Class:      EcalRecHitMiscalib
// 
/**\class EcalRecHitMiscalib EcalRecHitMiscalib.cc CalibCalorimetry/CaloMiscalibTools.src/EcalRecHitMiscalib.cc

 Description: Producer to miscalibrate (calibrated) Ecal RecHit 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Luca Malgeri
//         Created:  $Date: 2006/09/08 14:00:00 $
// $Id: EcalRecHitMiscalib.h,v 1.1 2006/09/08 14:42:06 malgeri Exp $
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

class EcalRecHitMiscalib : public edm::EDProducer {
   public:
      explicit EcalRecHitMiscalib(const edm::ParameterSet&);
      ~EcalRecHitMiscalib();


      virtual void produce(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

 std::string ecalHitsProducer_;
 std::string barrelHits_;
 std::string endcapHits_;
 std::string RecalibBarrelHits_;
 std::string RecalibEndcapHits_;
};

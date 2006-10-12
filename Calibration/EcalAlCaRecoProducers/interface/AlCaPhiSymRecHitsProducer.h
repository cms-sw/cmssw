// -*- C++ -*-
//
// Package:    AlCaPhiSymRecHitsProducer
// Class:      AlCaPhiSymRecHitsProducer
// 
/**\class AlCaPhiSymRecHitsProducer AlCaPhiSymRecHitsProducer.cc Calibration/EcalAlCaRecoProducers/src/AlCaPhiSymRecHitsProducer.cc

 Description: Producer for EcalRecHits to be used for phi-symmetry ECAL calibration

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  David Futyan
//         Created:  $Date$
// $Id$
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

class AlCaPhiSymRecHitsProducer : public edm::EDProducer {
   public:
      explicit AlCaPhiSymRecHitsProducer(const edm::ParameterSet&);
      ~AlCaPhiSymRecHitsProducer();


      virtual void produce(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

 std::string ecalHitsProducer_;
 std::string barrelHits_;
 std::string endcapHits_;
 std::string phiSymBarrelHits_;
 std::string phiSymEndcapHits_;
 double eCut_barl_;
 double eCut_endc_;  
};

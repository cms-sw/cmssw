// -*- C++ -*-
//
// Package:    AlCaElectronsProducer
// Class:      AlCaElectronsProducer
// 
/**\class AlCaElectronsProducer AlCaElectronsProducer.cc Calibration/EcalAlCaRecoProducers/src/AlCaElectronsProducer.cc

 Description: Example of a producer of AlCa electrons

 Implementation:
     <Notes on implementation>

*/
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Mon Jul 17 18:07:01 CEST 2006
// $Id: AlCaElectronsProducer.h,v 1.4 2006/09/25 16:49:22 meridian Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class AlCaElectronsProducer : public edm::EDProducer {
   public:
      explicit AlCaElectronsProducer(const edm::ParameterSet&);
      ~AlCaElectronsProducer();


      virtual void produce(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  
  edm::InputTag ebRecHitsLabel_;
  edm::InputTag electronLabel_;
  std::string alcaBarrelHitsCollection_;
  int etaSize_;
  int phiSize_;
  

};

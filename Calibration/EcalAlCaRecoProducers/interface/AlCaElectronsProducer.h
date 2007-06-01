#ifndef _ALCAELECTRONSPRODUCER_H
#define _ALCAELECTRONSPRODUCER_H

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
// $Id: AlCaElectronsProducer.h,v 1.7 2006/11/21 16:53:03 malgeri Exp $
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
  edm::InputTag eeRecHitsLabel_;
  edm::InputTag electronLabel_;
  std::string alcaBarrelHitsCollection_;
  std::string alcaEndcapHitsCollection_;
  int etaSize_;
  int phiSize_;
  

};

#endif

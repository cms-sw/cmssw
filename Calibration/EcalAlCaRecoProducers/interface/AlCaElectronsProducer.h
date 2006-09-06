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
// $Id: AlCaElectronsProducer.h,v 1.1 2006/07/24 10:03:26 lorenzo Exp $
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

class AlCaElectronsProducer : public edm::EDProducer {
   public:
      explicit AlCaElectronsProducer(const edm::ParameterSet&);
      ~AlCaElectronsProducer();


      virtual void produce(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------


 std::string pixelMatchElectronProducer_;
 std::string siStripElectronProducer_;
 
 std::string pixelMatchElectronCollection_;
 std::string siStripElectronCollection_;

 std::string alcaPixelMatchElectronCollection_;
 std::string alcaSiStripElectronCollection_;
 
 double ptCut_;  
};

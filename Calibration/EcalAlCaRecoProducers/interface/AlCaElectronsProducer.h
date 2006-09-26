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
// $Id: AlCaElectronsProducer.h,v 1.3 2006/09/13 15:20:29 lorenzo Exp $
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

 std::string BarrelHitsCollection_;
 std::string EndcapHitsCollection_;
 std::string ecalHitsProducer_;

 std::string basicClusterCollection_;
 std::string basicClusterProducer_;

// std::string islandBarrelBasicClusterCollection_;
// std::string islandBarrelBasicClusterProducer_;

// std::string islandBarrelSuperClusterCollection_;
// std::string islandBarrelSuperClusterProducer_;

// std::string correctedIslandBarrelSuperClusterCollection_;
 //std::string correctedIslandBarrelSuperClusterProducer_;

 std::string hybridSuperClusterCollection_;
 std::string hybridSuperClusterProducer_;
 
 std::string correctedHybridSuperClusterCollection_;
 std::string correctedHybridSuperClusterProducer_ ;

// output collections

 std::string alcaPixelMatchElectronCollection_;

 std::string alcaSiStripElectronCollection_;
 
 std::string alcaBarrelHitsCollection_;

// std::string alcaEndcapHitsCollection_;
 
 std::string alcaBasicClusterCollection_;
 
// std::string alcaIslandBarrelSuperClusterCollection_;
 
// std::string alcaCorrectedIslandBarrelSuperClusterCollection_;
 
 std::string alcaHybridSuperClusterCollection_;
 
 std::string alcaCorrectedHybridSuperClusterCollection_;
 
 double ptCut_;
   
// int evento;
};

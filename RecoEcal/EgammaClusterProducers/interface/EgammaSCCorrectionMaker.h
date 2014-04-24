#ifndef RecoEcal_EgammaClusterProducers_EgammaSCCorrectionMaker_h
#define RecoEcal_EgammaClusterProducers_EgammaSCCorrectionMaker_h

// -*- C++ -*-
//
// Package:    EgammaSCCorrectionMaker
// Class:      EgammaSCCorrectionMaker
// 
/**\class EgammaSCCorrectionMaker EgammaSCCorrectionMaker.cc EgammaSCCorrectionMaker/EgammaSCCorrectionMaker/src/EgammaSCCorrectionMaker.cc

 Description: Producer of corrected SuperClusters

*/
//
// Original Author:  Dave Evans
//         Created:  Thu Apr 13 15:50:17 CEST 2006
//
//

#include <memory>
#include <string>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include "RecoEcal/EgammaClusterAlgos/interface/EgammaSCEnergyCorrectionAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


class EgammaSCCorrectionMaker : public edm::stream::EDProducer<> {
	
   public:
     explicit EgammaSCCorrectionMaker(const edm::ParameterSet&);
     ~EgammaSCCorrectionMaker();
     virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

     EcalClusterFunctionBaseClass* energyCorrectionFunction_;
     EcalClusterFunctionBaseClass* crackCorrectionFunction_;
     EcalClusterFunctionBaseClass* localContCorrectionFunction_;


     // pointer to the correction algo object
     EgammaSCEnergyCorrectionAlgo *energyCorrector_;
    
     

     // vars for the correction algo
     bool applyEnergyCorrection_;
     bool applyCrackCorrection_;
     bool applyLocalContCorrection_;

     std::string energyCorrectorName_;
     std::string crackCorrectorName_;
     std::string localContCorrectorName_;

     int modeEB_;
     int modeEE_;

     //     bool oldEnergyScaleCorrection_;
     double sigmaElectronicNoise_;
     double etThresh_;
     
     // vars to get products
     edm::EDGetTokenT<EcalRecHitCollection>          rHInputProducer_;
     edm::EDGetTokenT<reco::SuperClusterCollection>  sCInputProducer_;
	 edm::InputTag rHTag_;

     reco::CaloCluster::AlgoId sCAlgo_;
     std::string outputCollection_;

};
#endif

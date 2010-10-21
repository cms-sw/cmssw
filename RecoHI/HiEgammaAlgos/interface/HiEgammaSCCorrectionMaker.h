#ifndef RecoHI_HiEgammaAlgos_HiEgammaSCCorrectionMaker_h
#define RecoHI_HiEgammaAlgos_HiEgammaSCCorrectionMaker_h

// -*- C++ -*-
//
// Package:    HiEgammaSCCorrectionMaker
// Class:      HiEgammaSCCorrectionMaker
// 
/**\class HiEgammaSCCorrectionMaker HiEgammaSCCorrectionMaker.cc HiEgammaSCCorrectionMaker/HiEgammaSCCorrectionMaker/src/HiEgammaSCCorrectionMaker.cc

 Description: Producer of corrected SuperClusters

*/
//
// Original Author:  Dave Evans
//         Created:  Thu Apr 13 15:50:17 CEST 2006
// $Id: HiEgammaSCCorrectionMaker.h,v 1.10 2009/03/29 02:25:55 ymaravin Exp $
//
//

#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include "RecoEcal/EgammaClusterAlgos/interface/EgammaSCEnergyCorrectionAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 

class HiEgammaSCCorrectionMaker : public edm::EDProducer {
	
   public:
     explicit HiEgammaSCCorrectionMaker(const edm::ParameterSet&);
     ~HiEgammaSCCorrectionMaker();
     virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

     EcalClusterFunctionBaseClass* EnergyCorrection_;

     // the debug level
     EgammaSCEnergyCorrectionAlgo::VerbosityLevel verbosity_;

     // pointer to the correction algo object
     EgammaSCEnergyCorrectionAlgo *energyCorrector_;
    
     // vars for the correction algo
     bool applyEnergyCorrection_;
     //     bool oldEnergyScaleCorrection_;
     double sigmaElectronicNoise_;
     double etThresh_;
     
     // vars to get products
     edm::InputTag rHInputProducer_;
     edm::InputTag sCInputProducer_;

     reco::CaloCluster::AlgoId sCAlgo_;
     std::string outputCollection_;

};
#endif

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
// $Id: HiEgammaSCCorrectionMaker.h,v 1.1 2010/11/04 18:32:48 edwenger Exp $
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

#include "RecoHI/HiEgammaAlgos/interface/HiEgammaSCEnergyCorrectionAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"


class HiEgammaSCCorrectionMaker : public edm::EDProducer {
	
   public:
     explicit HiEgammaSCCorrectionMaker(const edm::ParameterSet&);
     ~HiEgammaSCCorrectionMaker();
     virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

     EcalClusterFunctionBaseClass* EnergyCorrection_;

     // the debug level
     HiEgammaSCEnergyCorrectionAlgo::VerbosityLevel verbosity_;

     // pointer to the correction algo object
     HiEgammaSCEnergyCorrectionAlgo *energyCorrector_;
    
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
     edm::ESHandle<CaloTopology> theCaloTopo_;


};
#endif

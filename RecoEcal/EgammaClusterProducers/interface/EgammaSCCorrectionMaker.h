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
// $Id: EgammaSCCorrectionMaker.h,v 1.7 2008/03/03 11:33:52 kkaadze Exp $
//
//

#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoEcal/EgammaClusterAlgos/interface/EgammaSCEnergyCorrectionAlgo.h"

class EgammaSCCorrectionMaker : public edm::EDProducer {
	
   public:
     explicit EgammaSCCorrectionMaker(const edm::ParameterSet&);
     ~EgammaSCCorrectionMaker();
     virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

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

     reco::AlgoId sCAlgo_;
     std::string outputCollection_;

};
#endif

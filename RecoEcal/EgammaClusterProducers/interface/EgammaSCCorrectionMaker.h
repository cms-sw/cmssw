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
// $Id$
//
//

#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoEcal/EgammaClusterAlgos/interface/EgammaSCEnergyCorrectionAlgo.h"

class EgammaSCCorrectionMaker : public edm::EDProducer {
	
   public:
     explicit EgammaSCCorrectionMaker(const edm::ParameterSet&);
     ~EgammaSCCorrectionMaker();
     virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     EgammaSCEnergyCorrectionAlgo *energyCorrector_;
     
     bool applyEnergyCorrection_;
     double sigmaElectronicNoise_;
     std::string rHInputProducer_;
     std::string rHInputCollection_;     
     std::string sCInputProducer_;
     std::string sCInputCollection_;
     std::string outputCollection_;

};
#endif

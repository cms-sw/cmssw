// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTPhotonTrackIsolationProducers
// 
/**\class EgammaHLTPhotonTrackIsolationProducers EgammaHLTPhotonTrackIsolationProducers.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPhotonTrackIsolationProducers.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTPhotonTrackIsolationProducers.h,v 1.1 2006/10/24 13:50:39 monicava Exp $
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
// class declaration
//

class EgammaHLTPhotonTrackIsolationProducers : public edm::EDProducer {
   public:
      explicit EgammaHLTPhotonTrackIsolationProducers(const edm::ParameterSet&);
      ~EgammaHLTPhotonTrackIsolationProducers();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag trackProducer_;

  edm::ParameterSet conf_;

  double egTrkIsoPtMin_; 
  double egTrkIsoConeSize_;
  double egTrkIsoZSpan_;   
  double egTrkIsoRSpan_;  
  double egTrkIsoVetoConeSize_;

};


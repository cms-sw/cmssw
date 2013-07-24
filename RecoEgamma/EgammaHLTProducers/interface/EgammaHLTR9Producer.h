// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTR9Producer
// 
/**\class EgammaHLTR9Producer EgammaHLTR9Producer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTR9Producer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTR9Producer.h,v 1.2 2010/06/10 16:19:31 ghezzi Exp $
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

class EgammaHLTR9Producer : public edm::EDProducer {
   public:
      explicit EgammaHLTR9Producer(const edm::ParameterSet&);
      ~EgammaHLTR9Producer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag ecalRechitEBTag_;
  edm::InputTag ecalRechitEETag_;
  bool useSwissCross_;
  
  edm::ParameterSet conf_;

};


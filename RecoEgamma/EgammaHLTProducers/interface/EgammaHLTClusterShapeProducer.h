// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTClusterShapeProducer
// 
/**\class EgammaHLTClusterShapeProducer EgammaHLTClusterShapeProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTClusterShapeProducer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTClusterShapeProducer.h,v 1.2 2009/02/04 10:59:28 covarell Exp $
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

class EgammaHLTClusterShapeProducer : public edm::EDProducer {
   public:
      explicit EgammaHLTClusterShapeProducer(const edm::ParameterSet&);
      ~EgammaHLTClusterShapeProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag ecalRechitEBTag_;
  edm::InputTag ecalRechitEETag_;
  bool EtaOrIeta_;

  edm::ParameterSet conf_;

};


// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTR9IDProducer
// 
/**\class EgammaHLTR9IDProducer EgammaHLTR9IDProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTR9IDProducer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTR9IDProducer.h,v 1.1 2011/01/19 21:34:31 cgtully Exp $
//         modified by Chris Tully (Princeton)
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

class EgammaHLTR9IDProducer : public edm::EDProducer {
   public:
      explicit EgammaHLTR9IDProducer(const edm::ParameterSet&);
      ~EgammaHLTR9IDProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag ecalRechitEBTag_;
  edm::InputTag ecalRechitEETag_;
  
  edm::ParameterSet conf_;

};


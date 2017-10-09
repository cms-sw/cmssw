#ifndef RecoJets_JetProducers_plugins_CastorJetIDProducer_h
#define RecoJets_JetProducers_plugins_CastorJetIDProducer_h

// -*- C++ -*-
//
// Package:    CastorJetIDProducer
// Class:      CastorJetIDProducer
// 
/**\class CastorJetIDProducer CastorJetIDProducer.cc RecoJets/JetProducers/plugins/CastorJetIDProducer.cc

 Description: Produces a value map of jet---> jet Id

  
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Thu Sep 17 12:18:18 CDT 2009
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoJets/JetProducers/interface/CastorJetIDHelper.h"

//
// class decleration
//

class CastorJetIDProducer : public edm::stream::EDProducer<> {
   public:

      explicit CastorJetIDProducer(const edm::ParameterSet&);
      ~CastorJetIDProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      
      // ----------member data ---------------------------
      edm::InputTag                 src_;          // input jet source
      reco::helper::CastorJetIDHelper     helper_; // castor jet id helper algorithm

      edm::EDGetTokenT<edm::View<reco::BasicJet> > input_jet_token_;
};


#endif

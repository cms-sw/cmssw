#ifndef RecoJets_JetProducers_plugins_JetIDProducer_h
#define RecoJets_JetProducers_plugins_JetIDProducer_h

// -*- C++ -*-
//
// Package:    JetIDProducer
// Class:      JetIDProducer
// 
/**\class JetIDProducer JetIDProducer.cc RecoJets/JetProducers/plugins/JetIDProducer.cc

 Description: Produces a value map of jet---> jet Id

 Implementation:
     There are two modes: AOD only, in which case only a subset of the info is written,
     and RECO, when all the info is written. The AOD-only case will be suitable
     for the "very loose" jet ID, whereas the RECO case will be globally suitable. 
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
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoJets/JetProducers/interface/JetIDHelper.h"
#include "RecoJets/JetProducers/interface/JetMuonHitsIDHelper.h"

//
// class decleration
//

class JetIDProducer : public edm::EDProducer {
   public:

      explicit JetIDProducer(const edm::ParameterSet&);
      ~JetIDProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      
      // ----------member data ---------------------------
      edm::InputTag                 src_;         // input jet source
      reco::helper::JetIDHelper     helper_;      // jet id helper algorithm
      reco::helper::JetMuonHitsIDHelper muHelper_;    // jet id from muon rechits helper algorithm
      
      edm::EDGetTokenT<edm::View<reco::CaloJet> > input_jet_token_;

};


#endif

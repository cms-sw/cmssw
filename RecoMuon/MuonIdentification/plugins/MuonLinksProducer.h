#ifndef MuonIdentification_MuonLinksProducer_h
#define MuonIdentification_MuonLinksProducer_h

// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonLinksProducer
// 
/*
 Simple producer to make reco::MuonTrackLinks collection 
 out of the global muons from "muons" collection to restore
 dropped links used as input for MuonIdProducer.
 */
//
// Original Author:  Dmytro Kovalskyi
//
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class MuonLinksProducer : public edm::EDProducer {
 public:
   explicit MuonLinksProducer(const edm::ParameterSet&);
   
   virtual ~MuonLinksProducer();
   
   virtual void produce(edm::Event&, const edm::EventSetup&);
   
 private:
   edm::InputTag m_inputCollection;
   edm::EDGetTokenT<reco::MuonCollection> muonToken_; 

};
#endif

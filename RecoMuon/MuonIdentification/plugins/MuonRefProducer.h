#ifndef MuonIdentification_MuonRefProducer_h
#define MuonIdentification_MuonRefProducer_h

// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonRefProducer
// 
/*

 Description: create a reduced collection of muons based on a reference
              collection and a set of cuts.
*/
//
// Original Author:  Dmytro Kovalskyi
// $Id: MuonRefProducer.h,v 1.2 2007/05/15 18:31:05 jribnik Exp $
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMuon/MuonIdentification/interface/TrackerMuonIdentification.h"
class MuonRefProducer : public edm::EDProducer {
 public:
   explicit MuonRefProducer(const edm::ParameterSet&);
   virtual ~MuonRefProducer();
   virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
   bool goodMuon( const reco::Muon& );
   edm::InputTag theReferenceCollection;
   TrackerMuonIdentification theSelector;
};
#endif

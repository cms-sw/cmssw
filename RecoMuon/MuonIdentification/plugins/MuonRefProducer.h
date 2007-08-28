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
// $Id: MuonRefProducer.h,v 1.1 2007/07/16 23:57:32 dmytro Exp $
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "RecoMuon/MuonIdentification/interface/IdGlobalFunctions.h"

class MuonRefProducer : public edm::EDProducer {
 public:
   explicit MuonRefProducer(const edm::ParameterSet&);
   virtual ~MuonRefProducer();
   virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
   edm::InputTag theReferenceCollection_;

   muonid::AlgorithmType type_;
   int    minNumberOfMatches_;
   double maxAbsDx_;
   double maxAbsPullX_;
   double maxAbsDy_;
   double maxAbsPullY_;
   double maxChamberDist_;
   double maxChamberDistPull_;
   reco::Muon::ArbitrationType arbitrationType_;

};
#endif

#ifndef MuonTiming_MuonTimingProducer_h
#define MuonTiming_MuonTimingProducer_h 1

// -*- C++ -*-
//
// Package:    MuonTimingProducer
// Class:      MuonTimingProducer
// 
/**\class MuonTimingProducer MuonTimingProducer.h RecoMuon/MuonIdentification/interface/MuonTimingProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Piotr Traczyk, CERN
//         Created:  Mon Mar 16 12:27:22 CET 2009
// $Id$
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

#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "RecoMuon/MuonIdentification/plugins/DTTimingExtractor.h"


//
// class decleration
//

class MuonTimingProducer : public edm::EDProducer {
   public:
      explicit MuonTimingProducer(const edm::ParameterSet&);
      ~MuonTimingProducer();
      void fillTiming( reco::MuonRef muon, reco::MuonTimeExtra& dtTime, reco::MuonTimeExtra& cscTime, reco::MuonTimeExtra& combinedTime, edm::Event& iEvent, const edm::EventSetup& iSetup );

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      void fillTimeFromMeasurements( TimeMeasurementSequence tmSeq, reco::MuonTimeExtra &muTime );
      void rawFit(double &a, double &da, double &b, double &db, const vector<double> hitsx, const vector<double> hitsy);
      
      // ----------member data ---------------------------
      edm::InputTag m_muonCollection;

      DTTimingExtractor* theDTTimingExtractor_;

};

#endif

#ifndef MuonIdentification_MuonTimingFiller_h
#define MuonIdentification_MuonTimingFiller_h 1

// -*- C++ -*-
//
// Package:    MuonTimingFiller
// Class:      MuonTimingFiller
// 
/**\class MuonTimingFiller MuonTimingFiller.h RecoMuon/MuonIdentification/interface/MuonTimingFiller.h

 Description: Class filling the DT, CSC and Combined MuonTimeExtra objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Piotr Traczyk, CERN
//         Created:  Mon Mar 16 12:27:22 CET 2009
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
#include "RecoMuon/MuonIdentification/interface/DTTimingExtractor.h"
#include "RecoMuon/MuonIdentification/interface/CSCTimingExtractor.h"

//
// class decleration
//

class MuonTimingFiller {
   public:
  MuonTimingFiller(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
      ~MuonTimingFiller();
      void fillTiming( const reco::Muon& muon, reco::MuonTimeExtra& dtTime, 
                    reco::MuonTimeExtra& cscTime, reco::MuonTimeExtra& combinedTime, 
                    edm::Event& iEvent, const edm::EventSetup& iSetup );

   private:
      void fillTimeFromMeasurements( const TimeMeasurementSequence& tmSeq, reco::MuonTimeExtra &muTime );
      void rawFit(double &a, double &da, double &b, double &db, 
                  const std::vector<double>& hitsx, const std::vector<double>& hitsy);
      void addEcalTime( const reco::Muon& muon, TimeMeasurementSequence &cmbSeq );
      void combineTMSequences( const reco::Muon& muon, const TimeMeasurementSequence& dtSeq, 
                               const TimeMeasurementSequence& cscSeq, TimeMeasurementSequence &cmbSeq );
      
      DTTimingExtractor* theDTTimingExtractor_;
      CSCTimingExtractor* theCSCTimingExtractor_;
      double errorEB_,errorEE_,ecalEcut_;
      bool useDT_, useCSC_, useECAL_;

};

#endif

// -*- C++ -*-
//
// Package:    JetPlusTracks
// Class:      JetPlusTrackProducerAA
// 
/**\class JetPlusTrackProducerAA JetPlusTrackProducerAA.cc RecoJets/JetPlusTracks/src/JetPlusTrackProducerAA.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Olga Kodolova,40 R-A12,+41227671273,
//         Created:  Fri Feb 19 10:14:02 CET 2010
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
#include "RecoJets/JetPlusTracks/interface/JetPlusTrackCorrector.h"
#include "RecoJets/JetPlusTracks/interface/ZSPJPTJetCorrector.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"

#include <string>

//
// class declaration
//

class JetPlusTrackProducerAA : public edm::EDProducer {
   public:
      explicit JetPlusTrackProducerAA(const edm::ParameterSet&);
      ~JetPlusTrackProducerAA();
      virtual void beginJob();
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      reco::TrackRefVector calculateBGtracksJet(reco::JPTJetCollection&, std::vector <reco::TrackRef>&);
   private:
      
// Data      
      JetPlusTrackCorrector*        mJPTalgo;
      ZSPJPTJetCorrector*              mZSPalgo; 
      edm::InputTag                 src;
      edm::InputTag                 srcPVs_;
      std::string                   alias;
      bool                          vectorial_;  
      bool                          useZSP;
      edm::InputTag                 mTracks;
      double                        mConeSize;
      reco::TrackBase::TrackQuality trackQuality_;
      // ----------member data ---------------------------
};

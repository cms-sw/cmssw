// -*- C++ -*-
//
// Package:    JetPlusTracks
// Class:      JetPlusTrackProducer
// 
/**\class JetPlusTrackProducer JetPlusTrackProducer.cc JetPlusTrackProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Olga Kodolova,40 R-A12,+41227671273,
//         Created:  Fri Feb 19 10:14:02 CET 2010
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

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/TrackExtrapolation.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include <string>

//
// class declaration
//

class JetPlusTrackAddonSeedProducer : public edm::stream::EDProducer<> {
   public:
      explicit JetPlusTrackAddonSeedProducer(const edm::ParameterSet&);
      ~JetPlusTrackAddonSeedProducer() override;
      void produce(edm::Event&, const edm::EventSetup&) override;

   // ---------- private data members ---------------------------
   private:
      
      edm::InputTag          srcCaloJets;
      edm::InputTag          srcTrackJets;
      edm::InputTag          srcPVs_;
      
      std::string            alias;
      double                 ptCUT;
      bool                   usePAT;
      edm::EDGetTokenT<edm::View<reco::CaloJet> > input_jets_token_;
      edm::EDGetTokenT<edm::View<reco::TrackJet> > input_trackjets_token_;
      edm::EDGetTokenT<reco::VertexCollection> input_vertex_token_;  
      edm::EDGetTokenT<std::vector<pat::PackedCandidate> >   tokenPFCandidates_;
      edm::EDGetTokenT<CaloTowerCollection> input_ctw_token_; 

};

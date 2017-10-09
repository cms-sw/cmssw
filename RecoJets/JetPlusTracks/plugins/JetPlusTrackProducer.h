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

#include "JetPlusTrackCorrector.h"
#include "ZSPJPTJetCorrector.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


#include <string>

//
// class declaration
//

class JetPlusTrackProducer : public edm::stream::EDProducer<> {
   public:
      explicit JetPlusTrackProducer(const edm::ParameterSet&);
      ~JetPlusTrackProducer();
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

   // ---------- private data members ---------------------------
   private:
      
      JetPlusTrackCorrector* mJPTalgo;
      ZSPJPTJetCorrector*       mZSPalgo; 
      edm::InputTag          src;
      edm::InputTag          srcPVs_;
      std::string            alias;
      bool                   vectorial_;
      bool                   useZSP;
      double                 ptCUT;

      edm::EDGetTokenT<edm::View<reco::CaloJet> > input_jets_token_;
      edm::EDGetTokenT<reco::VertexCollection> input_vertex_token_;  
    
};

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
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/TrackExtrapolation.h"

#include <string>

//
// class declaration
//

class JetPlusTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit JetPlusTrackProducer(const edm::ParameterSet&);
  ~JetPlusTrackProducer() override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ---------- private data members ---------------------------
private:
  JetPlusTrackCorrector* mJPTalgo;
  ZSPJPTJetCorrector* mZSPalgo;
  edm::InputTag src_;
  edm::InputTag srcTrackJets_;
  edm::InputTag srcPVs_;
  std::string alias_;
  bool vectorial_;
  bool useZSP_;
  bool usePAT_;
  double ptCUT_;
  double dRcone_;

  edm::EDGetTokenT<edm::View<reco::CaloJet> > input_jets_token_;
  edm::EDGetTokenT<edm::View<reco::CaloJet> > input_addjets_token_;
  edm::EDGetTokenT<edm::View<reco::TrackJet> > input_trackjets_token_;
  edm::EDGetTokenT<reco::VertexCollection> input_vertex_token_;
  edm::EDGetTokenT<std::vector<reco::TrackExtrapolation> > mExtrapolations_;
};

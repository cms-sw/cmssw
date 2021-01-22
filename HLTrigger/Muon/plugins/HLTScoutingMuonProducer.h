#ifndef HLTScoutingMuonProducer_h
#define HLTScoutingMuonProducer_h

// -*- C++ -*-
//
// Package:    HLTrigger/Muon
// Class:      HLTScoutingMuonProducer
//
/**\class HLTScoutingMuonProducer HLTScoutingMuonProducer.h HLTScoutingMuonProducer.h

Description: Producer for Run3ScoutingMuon

*/
//
// Original Author:  David G. Sheffield (Rutgers)
//         Created:  Fri, 31 Jul 2015
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"

#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class HLTScoutingMuonProducer : public edm::global::EDProducer<> {
  typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::RecoChargedCandidate>, float, unsigned int>>
      RecoChargedCandMap;

public:
  explicit HLTScoutingMuonProducer(const edm::ParameterSet&);
  ~HLTScoutingMuonProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& setup) const final;

  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> ChargedCandidateCollection_;
  const edm::EDGetTokenT<reco::VertexCollection> displacedvertexCollection_;
  const edm::EDGetTokenT<reco::MuonCollection> MuonCollection_;
  const edm::EDGetTokenT<reco::MuonTrackLinksCollection> linkToken_;
  const edm::EDGetTokenT<reco::TrackCollection> TrackCollection_;
  const edm::EDGetTokenT<RecoChargedCandMap> EcalPFClusterIsoMap_;
  const edm::EDGetTokenT<RecoChargedCandMap> HcalPFClusterIsoMap_;
  const edm::EDGetTokenT<edm::ValueMap<double>> TrackIsoMap_;

  const double muonPtCut;
  const double muonEtaCut;
  const double minVtxProbCut;
};

#endif

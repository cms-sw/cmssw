// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonLinksProducer
//
//
// Original Author:  Dmytro Kovalskyi
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "RecoMuon/MuonIdentification/plugins/MuonLinksProducer.h"

#include <algorithm>

MuonLinksProducer::MuonLinksProducer(const edm::ParameterSet& iConfig) {
  produces<reco::MuonTrackLinksCollection>();
  m_inputCollection = iConfig.getParameter<edm::InputTag>("inputCollection");
  muonToken_ = consumes<reco::MuonCollection>(m_inputCollection);
}

MuonLinksProducer::~MuonLinksProducer() {}

void MuonLinksProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto output = std::make_unique<reco::MuonTrackLinksCollection>();
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muonToken_, muons);

  for (reco::MuonCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++muon) {
    if (!muon->isGlobalMuon())
      continue;
    output->push_back(reco::MuonTrackLinks(muon->track(), muon->standAloneMuon(), muon->combinedMuon()));
  }
  iEvent.put(std::move(output));
}

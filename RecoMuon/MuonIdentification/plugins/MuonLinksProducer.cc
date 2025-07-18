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
#include <algorithm>
#include <memory>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMuon/MuonIdentification/plugins/MuonLinksProducer.h"

MuonLinksProducer::MuonLinksProducer(const edm::ParameterSet& iConfig)
    : m_inputCollection{iConfig.getParameter<edm::InputTag>("inputCollection")},
      muonToken_{consumes<reco::MuonCollection>(m_inputCollection)} {
  produces<reco::MuonTrackLinksCollection>();
}

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

void MuonLinksProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputCollection", edm::InputTag("muons", "", "@skipCurrentProcess"));
  descriptions.addWithDefaultLabel(desc);
}

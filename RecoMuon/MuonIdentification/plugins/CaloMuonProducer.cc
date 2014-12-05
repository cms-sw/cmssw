// -*- C++ -*-
//
// Package:    CaloMuonProducer
// Class:      CaloMuonProducer
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Wed Oct  3 16:29:03 CDT 2007
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/MuonIdentification/plugins/CaloMuonProducer.h"

CaloMuonProducer::CaloMuonProducer(const edm::ParameterSet& iConfig)
{
   produces<reco::CaloMuonCollection>();
   inputCollection = iConfig.getParameter<edm::InputTag>("inputCollection");
   muonToken_ = consumes<reco::CaloMuonCollection>(inputCollection);
}

CaloMuonProducer::~CaloMuonProducer()
{
}

void CaloMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::Handle<reco::CaloMuonCollection> iMuons;
   iEvent.getByToken(muonToken_,iMuons);
   std::auto_ptr<reco::CaloMuonCollection> oMuons( new reco::CaloMuonCollection );
   for ( reco::CaloMuonCollection::const_iterator muon = iMuons->begin();
	 muon != iMuons->end(); ++muon )
     oMuons->push_back( *muon );
   iEvent.put(oMuons);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloMuonProducer);

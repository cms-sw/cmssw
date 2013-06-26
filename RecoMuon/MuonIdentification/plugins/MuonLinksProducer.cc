// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonLinksProducer
// 
//
// Original Author:  Dmytro Kovalskyi
// $Id: MuonLinksProducer.cc,v 1.2 2008/08/07 02:27:43 dmytro Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "RecoMuon/MuonIdentification/plugins/MuonLinksProducer.h"

#include <algorithm>

MuonLinksProducer::MuonLinksProducer(const edm::ParameterSet& iConfig)
{
   produces<reco::MuonTrackLinksCollection>();
   m_inputCollection = iConfig.getParameter<edm::InputTag>("inputCollection");
}

MuonLinksProducer::~MuonLinksProducer()
{
}

void MuonLinksProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   std::auto_ptr<reco::MuonTrackLinksCollection> output(new reco::MuonTrackLinksCollection());
   edm::Handle<reco::MuonCollection> muons; 
   iEvent.getByLabel(m_inputCollection, muons);
   
   for ( reco::MuonCollection::const_iterator muon = muons->begin(); 
	 muon != muons->end(); ++muon )
     {
	if ( ! muon->isGlobalMuon() ) continue;
	output->push_back( reco::MuonTrackLinks( muon->track(), muon->standAloneMuon(), muon->combinedMuon() ) );
     }
   iEvent.put( output );
}

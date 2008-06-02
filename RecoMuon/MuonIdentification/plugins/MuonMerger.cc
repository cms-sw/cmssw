// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonMerger
// 
//
// Original Author:  Dmytro Kovalskyi
// $Id: MuonMerger.cc,v 1.16 2007/10/31 21:44:41 dmytro Exp $
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
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "Utilities/Timing/interface/TimerStack.h"

#include <boost/regex.hpp>
#include "RecoMuon/MuonIdentification/plugins/MuonMerger.h"
#include "RecoMuon/MuonIdentification/interface/MuonIdTruthInfo.h"
#include "RecoMuon/MuonIdentification/interface/MuonArbitrationMethods.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include <algorithm>

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

MuonMerger::MuonMerger(const edm::ParameterSet& iConfig)
{
   produces<reco::MuonCollection>();
}


MuonMerger::~MuonMerger()
{
}

void MuonMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   std::auto_ptr<reco::MuonCollection> outputMuons(new reco::MuonCollection);
   edm::Handle<reco::MuonCollection> globalMuons, trackerMuons;
   iEvent.getByLabel("muons", globalMuons);
   iEvent.getByLabel("trackerMuons", trackerMuons);
   
   // global muons first
   for ( reco::MuonCollection::const_iterator muon = globalMuons->begin();
	 muon !=  globalMuons->end(); ++muon )
     outputMuons->push_back(*muon);
   
   // tracker muons next
   for ( reco::MuonCollection::const_iterator trackerMuon = trackerMuons->begin();
	 trackerMuon !=  trackerMuons->end(); ++trackerMuon )
     {
	// check if this muon is already in the list
	bool newMuon = true;
	for ( reco::MuonCollection::iterator muon = outputMuons->begin();
	      muon !=  outputMuons->end(); ++muon )
	  if ( muon->track().get() == trackerMuon->track().get() ) {
	     newMuon = false;
	     muon->setMatches( trackerMuon->getMatches() );
	     muon->setCalEnergy( trackerMuon->getCalEnergy() );
	     break;
	  }
	if ( newMuon ) outputMuons->push_back( *trackerMuon );
     }
   
   iEvent.put(outputMuons);
}

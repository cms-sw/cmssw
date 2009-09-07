// -*- C++ -*-
//
// Package:    CaloMuonProducer
// Class:      CaloMuonProducer
// 
/**\class CaloMuonProducer CaloMuonProducer.cc Test/CaloMuonProducer/src/CaloMuonProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Wed Oct  3 16:29:03 CDT 2007
// $Id: CaloMuonProducer.cc,v 1.3 2008/08/07 02:36:44 dmytro Exp $
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

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"
#include "RecoMuon/MuonIdentification/plugins/CaloMuonProducer.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

CaloMuonProducer::CaloMuonProducer(const edm::ParameterSet& iConfig)
{
   produces<reco::CaloMuonCollection>();
   inputMuons_ = iConfig.getParameter<edm::InputTag>("inputMuons");
   inputTracks_ = iConfig.getParameter<edm::InputTag>("inputTracks");
   caloCut_ = iConfig.getParameter<double>("minCaloCompatibility");
   minPt_ = iConfig.getParameter<double>("minPt");

   // Load TrackDetectorAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );
   
   // Load MuonCaloCompatibility parameters
   parameters = iConfig.getParameter<edm::ParameterSet>("MuonCaloCompatibility");
   muonCaloCompatibility_.configure( parameters );
}

CaloMuonProducer::~CaloMuonProducer()
{
}

reco::CaloMuon CaloMuonProducer::makeMuon( const edm::Event& iEvent, const edm::EventSetup& iSetup,
					   const reco::TrackRef& track )
{
   reco::CaloMuon aMuon;
   aMuon.setTrack( track );
   
   // propagation and association
   TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *(track.get()), parameters_);
   
   // energy
   reco::MuonEnergy muonEnergy;
   muonEnergy.em  = info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
   muonEnergy.had = info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
   muonEnergy.ho  = info.crossedEnergy(TrackDetMatchInfo::HORecHits);
   muonEnergy.emS9  = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits,1); // 3x3 energy
   muonEnergy.hadS9 = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits,1); // 3x3 energy
   muonEnergy.hoS9  = info.nXnEnergy(TrackDetMatchInfo::HORecHits,1);   // 3x3 energy
   aMuon.setCalEnergy( muonEnergy );
   
   // make a temporary reco::Muon to evaluate calo compatibility
   double energy = sqrt(track->p() * track->p() + 0.011163691);
   math::XYZTLorentzVector p4(track->px(), track->py(), track->pz(), energy);
   reco::Muon tmpMuon( track->charge(), p4, track->vertex() );
   tmpMuon.setCalEnergy( aMuon.calEnergy() );
   tmpMuon.setInnerTrack( aMuon.track() );
   
   // get calo compatibility
   aMuon.setCaloCompatibility( muonCaloCompatibility_.evaluate(tmpMuon) );
   return aMuon;
}


void CaloMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<reco::MuonCollection> muons;
   iEvent.getByLabel(inputMuons_,muons);
   Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(inputTracks_,tracks);

   std::auto_ptr<reco::CaloMuonCollection> caloMuons( new reco::CaloMuonCollection );
   
   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
   trackAssociator_.setPropagator(propagator.product());
   
   unsigned int i = 0;
   for ( reco::TrackCollection::const_iterator track = tracks->begin();
	 track != tracks->end(); ++track, ++i )
     {
	if (track->pt() < minPt_) continue;
	bool usedTrack = false;
	if ( muons.isValid() )
	  for ( reco::MuonCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++muon )
	    if ( muon->track().get() == &*track )
	      {
		 usedTrack = true;
		 break;
	      }
	if ( usedTrack ) continue;
	reco::CaloMuon caloMuon( makeMuon( iEvent, iSetup, reco::TrackRef( tracks, i ) ) );
	if ( ! caloMuon.isCaloCompatibilityValid() || caloMuon.caloCompatibility() < caloCut_ ) continue;
	caloMuons->push_back( caloMuon );
     }
   iEvent.put(caloMuons);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloMuonProducer);

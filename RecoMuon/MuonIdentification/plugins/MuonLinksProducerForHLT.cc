/** \class MuonLinksProducerForHLT
 *
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "RecoMuon/MuonIdentification/plugins/MuonLinksProducerForHLT.h"

//#include <algorithm>

MuonLinksProducerForHLT::MuonLinksProducerForHLT(const edm::ParameterSet& iConfig)
{
   produces<reco::MuonTrackLinksCollection>();
   theLinkCollectionInInput = iConfig.getParameter<edm::InputTag>("LinkCollection");
   theInclusiveTrackCollectionInInput = iConfig.getParameter<edm::InputTag>("InclusiveTrackerTrackCollection");
   ptMin = iConfig.getParameter<double>("ptMin");
   pMin = iConfig.getParameter<double>("pMin");
   shareHitFraction = iConfig.getParameter<double>("shareHitFraction");

   linkToken_ = consumes<reco::MuonTrackLinksCollection>(theLinkCollectionInInput);
   trackToken_ = consumes<reco::TrackCollection>(theInclusiveTrackCollectionInInput);
   
}

MuonLinksProducerForHLT::~MuonLinksProducerForHLT()
{
}

void MuonLinksProducerForHLT::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
   std::auto_ptr<reco::MuonTrackLinksCollection> output(new reco::MuonTrackLinksCollection());

   edm::Handle<reco::MuonTrackLinksCollection> links; 
   iEvent.getByToken(linkToken_, links);

   edm::Handle<reco::TrackCollection> incTracks; 
   iEvent.getByToken(trackToken_, incTracks);

   for(reco::MuonTrackLinksCollection::const_iterator link = links->begin(); 
       link != links->end(); ++link){
     bool found = false;
     unsigned int trackIndex = 0;
     unsigned int muonTrackHits = link->trackerTrack()->extra()->recHitsSize();
     for(reco::TrackCollection::const_iterator track = incTracks->begin();
	 track != incTracks->end(); ++track, ++trackIndex){      
       if ( track->pt() < ptMin ) continue;
       if ( track->p() < pMin ) continue;
       //std::cout << "pt (muon/track) " << link->trackerTrack()->pt() << " " << track->pt() << std::endl;
       unsigned trackHits = track->extra()->recHitsSize();
       //std::cout << "hits (muon/track) " << muonTrackHits  << " " << trackHits() << std::endl;
       unsigned int smallestNumberOfHits = trackHits < muonTrackHits ? trackHits : muonTrackHits;
       int numberOfCommonDetIds = 0;
       for ( auto hit = track->extra()->recHitsBegin();
	     hit != track->extra()->recHitsEnd(); ++hit ) {
	 for ( auto mit = link->trackerTrack()->extra()->recHitsBegin();
	     mit != link->trackerTrack()->extra()->recHitsEnd(); ++mit ) {
	   if ( (*hit)->geographicalId() == (*mit)->geographicalId() && 
		(*hit)->sharesInput((*mit),TrackingRecHit::some) ) { 
	     numberOfCommonDetIds++;
	     break;
	   }
	 }
       }
       double fraction = (double)numberOfCommonDetIds/smallestNumberOfHits;
       // std::cout << "Overlap/Smallest/fraction = " << numberOfCommonDetIds << " " << smallestNumberOfHits << " " << fraction << std::endl;
       if( fraction > shareHitFraction ) { 
	 output->push_back(reco::MuonTrackLinks(reco::TrackRef(incTracks,trackIndex), 
						link->standAloneTrack(), 
						link->globalTrack() ) );
	 found = true;
	 break;
       }
     }
     if (!found) 
       output->push_back(reco::MuonTrackLinks(link->trackerTrack(), 
					      link->standAloneTrack(), 
					      link->globalTrack() ) );
   }  
   iEvent.put( output );
}

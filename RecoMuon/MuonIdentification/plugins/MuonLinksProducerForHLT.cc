/** \class MuonLinksProducerForHLT
 *
 *  $Date: 2011/05/02 16:09:31 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
//#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "RecoMuon/MuonIdentification/plugins/MuonLinksProducerForHLT.h"

//#include <algorithm>

MuonLinksProducerForHLT::MuonLinksProducerForHLT(const edm::ParameterSet& iConfig)
{
   produces<reco::MuonTrackLinksCollection>();
   theLinkCollectionInInput = iConfig.getParameter<edm::InputTag>("LinkCollection");
   theInclusiveTrackCollectionInInput = iConfig.getParameter<edm::InputTag>("InclusiveTrackerTrackCollection");
}

MuonLinksProducerForHLT::~MuonLinksProducerForHLT()
{
}

void MuonLinksProducerForHLT::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   std::auto_ptr<reco::MuonTrackLinksCollection> output(new reco::MuonTrackLinksCollection());

   edm::Handle<reco::MuonTrackLinksCollection> links; 
   iEvent.getByLabel(theLinkCollectionInInput, links);

   edm::Handle<reco::TrackCollection> incTracks; 
   iEvent.getByLabel(theInclusiveTrackCollectionInInput, incTracks);

   for(reco::MuonTrackLinksCollection::const_iterator link = links->begin(); 
       link != links->end(); ++link){
     bool found = false;
     unsigned int trackIndex = 0;
     for(reco::TrackCollection::const_iterator track = incTracks->begin();
	 track != incTracks->end(); ++track, ++trackIndex){      
       if(found) continue;
       if(track->momentum() == link->trackerTrack()->momentum()){
	 output->push_back(reco::MuonTrackLinks(reco::TrackRef(incTracks,trackIndex), 
						link->standAloneTrack(), 
						link->globalTrack() ) );
	 found = true;
       }
     }
     if (!found) 
       output->push_back(reco::MuonTrackLinks(link->trackerTrack(), 
					      link->standAloneTrack(), 
					      link->globalTrack() ) );
   }  
   iEvent.put( output );
}

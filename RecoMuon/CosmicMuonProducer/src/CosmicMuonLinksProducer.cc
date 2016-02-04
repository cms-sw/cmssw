#include "RecoMuon/CosmicMuonProducer/src/CosmicMuonLinksProducer.h"

/**\class CosmicMuonLinksProducer
 *
 *
 * $Date: 2009/01/05 21:58:19 $
 * $Revision: 1.2 $
 * Original Author:  Chang Liu - Purdue University <chang.liu@cern.ch>
**/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

using namespace edm;
using namespace std;
using namespace reco;

//
// constructors and destructor
//
CosmicMuonLinksProducer::CosmicMuonLinksProducer(const ParameterSet& iConfig)
{

  category_ = "Muon|RecoMuon|CosmicMuon|CosmicMuonLinksProducer";

  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");

  theService = new MuonServiceProxy(serviceParameters);

  std::vector<edm::ParameterSet> theMapPSets = iConfig.getParameter<std::vector<edm::ParameterSet> >("Maps");
  for (std::vector<edm::ParameterSet>::const_iterator iMPS = theMapPSets.begin();
       iMPS != theMapPSets.end(); iMPS++) {
     edm::InputTag subTrackTag = (*iMPS).getParameter<edm::InputTag>("subTrack");
     edm::InputTag parentTrackTag = (*iMPS).getParameter<edm::InputTag>("parentTrack");
     theTrackLinks.push_back( make_pair(subTrackTag, parentTrackTag) );
  }

  for(std::vector<std::pair<edm::InputTag, edm::InputTag> >::const_iterator iLink = theTrackLinks.begin();
     iLink != theTrackLinks.end(); iLink++) {
    LogDebug(category_) << "preparing map between " << (*iLink).first<<" & "<< (*iLink).second;
    std::string mapname = (*iLink).first.label() + "To" + (*iLink).second.label();
    produces<reco::TrackToTrackMap>(mapname);
  }

}

CosmicMuonLinksProducer::~CosmicMuonLinksProducer()
{
  if (theService) delete theService;
}


// ------------ method called to produce the data  ------------
void
CosmicMuonLinksProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  LogInfo(category_) << "Processing event number: " << iEvent.id();

  theService->update(iSetup);

  for(std::vector<std::pair<edm::InputTag, edm::InputTag> >::const_iterator iLink = theTrackLinks.begin();
     iLink != theTrackLinks.end(); iLink++){
    LogDebug(category_) << "making map between " << (*iLink).first<<" and "<< (*iLink).second;
    std::string mapname = (*iLink).first.label() + "To" + (*iLink).second.label();
    reco::TrackToTrackMap ttmap;

    Handle<reco::TrackCollection> subTracks;
    Handle<reco::TrackCollection> parentTracks;

    if ( iEvent.getByLabel( (*iLink).first, subTracks) && iEvent.getByLabel( (*iLink).second, parentTracks) ) {

	 ttmap = mapTracks(subTracks, parentTracks); 
         LogTrace(category_) << "Mapped: "<<
(*iLink).first.label()<<" "<<subTracks->size()<< " and "<<(*iLink).second.label()<<" "<<parentTracks->size()<<", results: "<< ttmap.size() <<endl;

    }

    auto_ptr<reco::TrackToTrackMap> trackToTrackmap(new reco::TrackToTrackMap(ttmap));
    iEvent.put(trackToTrackmap, mapname);
  }

}

reco::TrackToTrackMap CosmicMuonLinksProducer::mapTracks(const Handle<reco::TrackCollection>& subTracks, const Handle<reco::TrackCollection>& parentTracks) const {
  reco::TrackToTrackMap map;
  for ( unsigned int position1 = 0; position1 != subTracks->size(); ++position1) {
    TrackRef track1(subTracks, position1);
    for ( unsigned int position2 = 0; position2 != parentTracks->size(); ++position2) {
      TrackRef track2(parentTracks, position2);
      int shared = sharedHits(*track1, *track2); 
      LogTrace(category_)<<"sharedHits "<<shared<<" track1 "<<track1->found()<<" track2 "<<track2->found()<<endl;
 
      if (shared > (track1->found())/2 ) map.insert(track1, track2);
    }
  }

  return map; 
}

int CosmicMuonLinksProducer::sharedHits(const reco::Track& track1, const reco::Track& track2) const {

  int match = 0;

  for (trackingRecHit_iterator hit1 = track1.recHitsBegin(); hit1 != track1.recHitsEnd(); ++hit1) {
    if ( !(*hit1)->isValid() ) continue;
    DetId id1 = (*hit1)->geographicalId();
    if ( id1.det() != DetId::Muon ) continue; //ONLY MUON
    LogTrace(category_)<<"first ID "<<id1.rawId()<<" "<<(*hit1)->localPosition()<<endl;
    GlobalPoint pos1 = theService->trackingGeometry()->idToDet(id1)->surface().toGlobal((*hit1)->localPosition());

    for (trackingRecHit_iterator hit2 = track2.recHitsBegin(); hit2 != track2.recHitsEnd(); ++hit2) {

          if ( !(*hit2)->isValid() ) continue;

          DetId id2 = (*hit2)->geographicalId();
          if ( id2.det() != DetId::Muon ) continue; //ONLY MUON

//          LogTrace(category_)<<"second ID "<<id2.rawId()<< (*hit2)->localPosition()<<endl;

          if (id2.rawId() != id1.rawId() ) continue;

          GlobalPoint pos2 = theService->trackingGeometry()->idToDet(id2)->surface().toGlobal((*hit2)->localPosition());
	  if ( ( pos1 - pos2 ).mag()< 10e-5 ) match++;

        }

    }

   return match;

}

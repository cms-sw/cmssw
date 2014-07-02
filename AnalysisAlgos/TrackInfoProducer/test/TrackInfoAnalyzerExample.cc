#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"


#include <iostream>
#include <string>

using namespace edm;

class TrackInfoAnalyzerExample : public edm::EDAnalyzer {
  edm::EDGetTokenT<reco::TrackInfoTrackAssociationCollection> TItkassociatorCollectionToken;
  edm::EDGetTokenT<reco::TrackCollection> tkCollectionToken;
 public:
  TrackInfoAnalyzerExample(const edm::ParameterSet& pset);

  ~TrackInfoAnalyzerExample(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){

    using namespace reco;

    //get TrackInfoTrackAssociationCollection from the event
    edm::Handle<reco::TrackInfoTrackAssociationCollection> TItkassociatorCollection;
    event.getByToken(TItkassociatorCollectionToken,TItkassociatorCollection);

    // get track collection from the event
    edm::Handle<reco::TrackCollection> tkCollection;
    event.getByToken(tkCollectionToken,tkCollection);
    edm::LogInfo("TrackInfoAnalyzerExample")<<"track info collection size "<<TItkassociatorCollection->size();
    edm::LogInfo("TrackInfoAnalyzerExample")<<"track collection size "<<tkCollection->size();

    // loop on the tracks
    for (unsigned int track=0;track<tkCollection->size();++track){

      //build the ref to the track
      reco::TrackRef trackref=reco::TrackRef(tkCollection,track);
      edm::LogInfo("TrackInfoAnalyzerExample")<<"Track pt"<<trackref->pt();

      //get the ref to the trackinfo
      reco::TrackInfoRef trackinforef=(*TItkassociatorCollection.product())[trackref];

      // get additional track information from trackinfo:

      //the seed:
      const TrajectorySeed seed=trackinforef->seed();
      edm::LogInfo("TrackInfoAnalyzerExample") <<"N hits in the seed: "<<seed.nHits();
      edm::LogInfo("TrackInfoAnalyzerExample") <<"Starting state position"<<seed.startingState().parameters().position();
      edm::LogInfo("TrackInfoAnalyzerExample") <<"Starting state direction"<<seed.startingState().parameters().momentum();


      //local angle for a specific hit
      TrackingRecHitRef rechitref=trackref->recHit(2);
      if(rechitref->isValid()){
	const LocalVector localdir=trackinforef->localTrackMomentum(Combined,rechitref);
	edm::LogInfo("TrackInfoAnalyzerExample") <<"Local x-z plane angle of 3rd hit:"<<atan2(localdir.x(),localdir.z());
      }

      // loop on all the track hits
      reco::TrackInfo::TrajectoryInfo::const_iterator iter;
      for(iter=trackinforef->trajStateMap().begin();iter!=trackinforef->trajStateMap().end();iter++){

	//trajectory local direction and position on detector
	LocalVector statedirection=(trackinforef->stateOnDet(Combined,(*iter).first)->parameters()).momentum();
	LocalPoint  stateposition=(trackinforef->stateOnDet(Combined,(*iter).first)->parameters()).position();
	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalMomentum: "<<statedirection;
	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalPosition: "<<stateposition;
	edm::LogInfo("TrackInfoAnalyzerExample") <<"Local x-z plane angle: "<<atan2(statedirection.x(),statedirection.z());
	if(trackinforef->type((*iter).first)==Matched){ // get the direction for the components
	  edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalMomentum (mono): "<<trackinforef->localTrackMomentumOnMono(Combined,(*iter).first);
	  edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalMomentum (stereo): "<<trackinforef->localTrackMomentumOnStereo(Combined,(*iter).first);
	}
	else if (trackinforef->type((*iter).first)==Projected){//one should be 0
	  edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalMomentum (mono): "<<trackinforef->localTrackMomentumOnMono(Combined,(*iter).first);
	  edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalMomentum (stereo): "<<trackinforef->localTrackMomentumOnStereo(Combined,(*iter).first);
	}
	//hit position on detector
	if(((*iter).first)->isValid())edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalPosition (rechit): "<<((*iter).first)->localPosition();
      }
    }
  }

};

TrackInfoAnalyzerExample::TrackInfoAnalyzerExample(const edm::ParameterSet& pset)
: TItkassociatorCollectionToken(consumes<reco::TrackInfoTrackAssociationCollection>(pset.getParameter<edm::InputTag>("TrackInfo")))
, tkCollectionToken(consumes<reco::TrackCollection>(pset.getParameter<edm::InputTag>("Tracks")))
{}


DEFINE_FWK_MODULE(TrackInfoAnalyzerExample);


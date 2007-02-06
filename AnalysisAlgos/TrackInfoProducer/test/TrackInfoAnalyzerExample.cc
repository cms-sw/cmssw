#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <iostream>
#include <string>

using namespace edm;

class TrackInfoAnalyzerExample : public edm::EDAnalyzer {
 public:
  TrackInfoAnalyzerExample(const edm::ParameterSet& pset) {conf_=pset;}

  ~TrackInfoAnalyzerExample(){}
  edm::ParameterSet conf_;

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){

    using namespace std;

    //std::cout << "\nEvent ID = "<< event.id() << std::endl ;
    edm::InputTag TkiTag = conf_.getParameter<edm::InputTag>("TrackInfo");
    edm::Handle<reco::TrackInfoTrackAssociationCollection> TItkassociatorCollection;
    event.getByLabel(TkiTag,TItkassociatorCollection);
    edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("Tracks");
    edm::Handle<reco::TrackCollection> tkCollection;
    event.getByLabel(TkTag,tkCollection);
      

    //    edm::LogInfo("TrackInfoAnalyzerExample") <<"number of infos "<< tC.size();
 
    //    for (reco::TrackCollection::const_iterator track=tkCollection->begin(); track!=tkCollection->end(); ++track){
    for (unsigned int track=0;track<tkCollection->size();++track){
      reco::TrackRef trackref=reco::TrackRef(tkCollection,track);
      edm::LogInfo("TrackInfoAnalyzerExample")<<"Track pt"<<trackref->pt();
      //const reco::TrackInfo::TrajectoryInfo tinfo=track->trajstate();
      reco::TrackInfo::TrajectoryInfo::const_iterator iter;
      reco::TrackInfoRef trackinforef=(*TItkassociatorCollection.product())[trackref];
      edm::LogInfo("TrackInfoAnalyzerExample") <<"N hits in the seed: "<<(*TItkassociatorCollection.product())[trackref]->seed().nHits();
      //      edm::LogInfo("TrackInfoAnalyzerExample") <<"Starting state "<<track->second->seed().startingState().parameters().position();
      // loop on the track hits
      for(iter=(*TItkassociatorCollection.product())[trackref]->trajStateMap().begin();iter!=(*TItkassociatorCollection.product())[trackref]->trajStateMap().end();iter++){
	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalMomentum: "<<((*iter).second.parameters()).momentum();
	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalPosition: "<<((*iter).second.parameters()).position();
	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalPosition (rechit): "<<((*iter).first)->localPosition();
      }
    }
  }
};

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TrackInfoAnalyzerExample);


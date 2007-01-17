#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
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
    edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("TrackInfo");
    edm::Handle<reco::TrackInfoCollection> trackCollection;
    event.getByLabel(TkTag, trackCollection);
      
    reco::TrackInfoCollection tC = *(trackCollection.product());

    edm::LogInfo("TrackInfoAnalyzerExample") <<"number of infos "<< tC.size();
    for (reco::TrackInfoCollection::iterator track=tC.begin(); track!=tC.end(); track++){
      
      //const reco::TrackInfo::TrajectoryInfo tinfo=track->trajstate();
      reco::TrackInfo::TrajectoryInfo::const_iterator iter;
      edm::LogInfo("TrackInfoAnalyzerExample") <<"N hits in the seed: "<<track->seed().nHits();
      edm::LogInfo("TrackInfoAnalyzerExample") <<"Starting stare "<<track->seed().startingState().parameters().position();
      for(iter=track->trajStateMap().begin();iter!=track->trajStateMap().end();iter++){
	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalMomentum: "<<((*iter).second.parameters()).momentum();
	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalPosition: "<<((*iter).second.parameters()).position();
	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalPosition (rechit): "<<((*iter).first)->localPosition();
      }
    }
  }
};

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TrackInfoAnalyzerExample);


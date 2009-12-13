// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"


using namespace std;
using namespace edm;
using namespace reco;


class TrackFilter : public EDFilter {
 public:
  explicit TrackFilter(const edm::ParameterSet& pset);
  ~TrackFilter();
  virtual bool filter(edm::Event& event, const edm::EventSetup& eventSetup);
       
protected:
       
private: 

  InputTag theTrackTag;
  unsigned int theMinNum;

};


TrackFilter::TrackFilter(const ParameterSet& pset){
  // input tags

  theTrackTag =  pset.getParameter<edm::InputTag>("trackLabel");
  theMinNum   =  pset.getParameter<unsigned int>("atLeastNTracks");
}

TrackFilter::~TrackFilter(){
}

bool TrackFilter::filter(edm::Event& event, const edm::EventSetup& eventSetup){

 const std::string metname = "Muon|MuonAnalysis|TrackFilter";

 Handle<reco::TrackCollection> tracks;
 event.getByLabel(theTrackTag,tracks);
 LogTrace(metname) << "Number of tracks " << tracks->size() << endl;
 
 return (tracks->size() >= theMinNum);
}

DEFINE_FWK_MODULE(TrackFilter);

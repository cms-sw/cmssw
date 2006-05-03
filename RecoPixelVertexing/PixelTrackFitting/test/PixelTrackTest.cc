#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <iostream>

using namespace std;

class PixelTrackTest : public edm::EDAnalyzer {
public:
  explicit PixelTrackTest(const edm::ParameterSet& conf);
  ~PixelTrackTest();
  virtual void beginJob(const edm::EventSetup& es) { }
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void endJob() { }
private:
  edm::ParameterSet conf_; 
};

PixelTrackTest::PixelTrackTest(const edm::ParameterSet& conf)
  : conf_(conf)
{
  edm::LogInfo("PixelTrackTest")<<" CTOR";
}

PixelTrackTest::~PixelTrackTest()
{
  edm::LogInfo("PixelTrackTest")<<" DTOR";
}

void PixelTrackTest::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{
  cout <<"*** PixelTrackTest, analyze event: " << ev.id() << endl;
  edm::Handle<reco::TrackCollection> trackCollection;
  std::string trackCollName = conf_.getParameter<std::string>("TrackCollection");
  ev.getByLabel(trackCollName,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());

  std::cout << *(trackCollection.provenance()) << std::endl;
  cout << "Reconstructed "<< tracks.size() << " tracks" << std::endl;
  typedef reco::TrackCollection::const_iterator IT;
  for (IT track=tracks.begin(); track!=tracks.end(); track++){
    cout << "\tmomentum: " << track->momentum()
         << "\tPT: " << track->pt()<< endl;
    cout << "\tvertex: " << track->vertex()
         << "\timpact parameter: " << track->d0()<< endl;
    cout << "\tcharge: " << track->charge()<< endl;
//    cout <<"\t\tNumber of RecHits "<<track->recHitsSize()<<endl;
  cout <<"------------------------------------------------"<<endl;
  }

}
 
DEFINE_FWK_MODULE(PixelTrackTest)

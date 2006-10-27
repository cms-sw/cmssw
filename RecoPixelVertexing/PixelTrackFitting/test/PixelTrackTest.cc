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
#include "DataFormats/TrackReco/interface/print.h"

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
  string collectionLabel;
  void myprint(const reco::Track & track) const;
};

PixelTrackTest::PixelTrackTest(const edm::ParameterSet& conf)
{
  collectionLabel = conf.getParameter<std::string>("TrackCollection");
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
  typedef reco::TrackCollection::const_iterator IT;

  edm::Handle<reco::TrackCollection> trackCollection;
  ev.getByLabel(collectionLabel,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());
  cout << "Number of tracks: "<< tracks.size() << " tracks" << std::endl;
  for (IT it=tracks.begin(); it!=tracks.end(); it++) myprint(*it);

  cout <<"------------------------------------------------"<<endl;
}

void PixelTrackTest::myprint(const reco::Track & track) const
{
    cout << "--- RECONSTRUCTED TRACK: " << endl;
    cout << "\tmomentum: " << track.momentum()
         << "\tPT: " << track.pt()<< endl;
    cout << "\tvertex: " << track.vertex()
         << "\t zip: " <<  track.dz()<<"+/-"<<track.dzError()
         << "\t tip: " << track.d0()<<"+/-"<<track.d0Error()<< endl;
    cout << "\t chi2: "<< track.chi2()<<endl;
    cout << "\tcharge: " << track.charge()<< endl;
}

 
DEFINE_FWK_MODULE(PixelTrackTest);

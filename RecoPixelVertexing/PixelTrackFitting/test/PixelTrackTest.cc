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
  void myprint(const reco::Track & track) const;
};

PixelTrackTest::PixelTrackTest(const edm::ParameterSet& conf)
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
//  edm::Handle<reco::TrackCollection> trackCollection1;
// ev.getByType(trackCollection);

  typedef reco::TrackCollection::const_iterator IT;
  edm::Handle<reco::TrackCollection> trackCollection1;
  ev.getByLabel("tracks1",trackCollection1);
  const reco::TrackCollection tracks1 = *(trackCollection1.product());

  cout << "Reconstructed TRACKS1: "<< tracks1.size() << " tracks" << std::endl;
  for (IT it=tracks1.begin(); it!=tracks1.end(); it++) myprint(*it);

  edm::Handle<reco::TrackCollection> trackCollection2;
  ev.getByLabel("tracks2",trackCollection2);
  const reco::TrackCollection tracks2 = *(trackCollection2.product());
  cout << "Reconstructed TRACKS2: "<< tracks2.size() << " tracks" << std::endl;
  for (IT it=tracks2.begin(); it!=tracks2.end(); it++) myprint(*it);

  cout <<"------------------------------------------------"<<endl;
}

void PixelTrackTest::myprint(const reco::Track & track) const
{
    cout << "--- RECONSTRUCTED TRACK: " << endl;
    cout << "\tmomentum: " << track.momentum()
         << "\tPT: " << track.pt()<< endl;
    cout << "\tvertex: " << track.vertex()
         << "\timpact parameter: " << track.d0()<< endl;
    cout << "\tcharge: " << track.charge()<< endl;
//    cout <<"\t\tNumber of RecHits "<<track->recHitsSize()<<endl;
//  cout <<"PRINT: " << print(*track) << endl;

}

 
DEFINE_FWK_MODULE(PixelTrackTest)

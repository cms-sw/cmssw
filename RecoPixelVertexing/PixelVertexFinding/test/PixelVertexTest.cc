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

#include "RecoPixelVertexing/PixelVertexFinding/interface/PVPositionBuilder.h"

#include <iostream>

using namespace std;

class PixelVertexTest : public edm::EDAnalyzer {
public:
  explicit PixelVertexTest(const edm::ParameterSet& conf);
  ~PixelVertexTest();
  virtual void beginJob(const edm::EventSetup& es) { }
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void endJob() { }
private:
  edm::ParameterSet conf_; 
};

PixelVertexTest::PixelVertexTest(const edm::ParameterSet& conf)
  : conf_(conf)
{
  edm::LogInfo("PixelVertexTest")<<" CTOR";
}

PixelVertexTest::~PixelVertexTest()
{
  edm::LogInfo("PixelVertexTest")<<" DTOR";
}

void PixelVertexTest::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{
  cout <<"*** PixelVertexTest, analyze event: " << ev.id() << endl;
  edm::Handle<reco::TrackCollection> trackCollection;
  std::string trackCollName = conf_.getParameter<std::string>("TrackCollection");
  ev.getByLabel(trackCollName,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());

  std::vector<const reco::Track* >trks;

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
    trks.push_back( &(*track) );
  cout <<"------------------------------------------------"<<endl;
  }
  PVPositionBuilder pos;  
  std::cout << "The average z-position of these tracks is " << pos.average(trks).value() << std::endl;
  std::cout << "The weighted average z-position of these tracks is " << pos.wtAverage(trks).value() << std::endl;
  
}
 
DEFINE_FWK_MODULE(PixelVertexTest)

#include "PixelVertexProducerMedian.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <fstream>
#include <vector>
#include <algorithm>

/*****************************************************************************/
struct ComparePairs
{
  bool operator() (const reco::Track * t1,
                   const reco::Track * t2)
  {
    return (t1->vz() < t2->vz());
  };
};

/*****************************************************************************/
PixelVertexProducerMedian::PixelVertexProducerMedian
  (const edm::ParameterSet& ps) : theConfig(ps)
{
  produces<reco::VertexCollection>();
}


/*****************************************************************************/
PixelVertexProducerMedian::~PixelVertexProducerMedian()
{ 
}

/*****************************************************************************/
void PixelVertexProducerMedian::beginJob
  (const edm::EventSetup& es)
{
}

/*****************************************************************************/
void PixelVertexProducerMedian::produce
  (edm::Event& ev, const edm::EventSetup& es)
{
  // Get pixel tracks
  edm::Handle<reco::TrackCollection> trackCollection;
  std::string trackCollectionName =
    theConfig.getParameter<std::string>("TrackCollection");
  ev.getByLabel(trackCollectionName, trackCollection);
  const reco::TrackCollection tracks_ = *(trackCollection.product());

  thePtMin = theConfig.getParameter<double>("PtMin");

  // Select tracks 
  std::vector<const reco::Track *> tracks;
  for (unsigned int i=0; i<tracks_.size(); i++)
  {
    if (tracks_[i].pt() > thePtMin)
    {
      reco::TrackRef recTrack(trackCollection, i);
      tracks.push_back( &(*recTrack));
    }
  }

  std::cerr << " [VertexProducer] selected tracks "
            << tracks.size() << " (" << tracks_.size() << ")" << std::endl; 

  // Sort along vertex z position
  std::sort(tracks.begin(), tracks.end(), ComparePairs());
  
  float vz;
  if(tracks.size() % 2 == 0)
    vz = (tracks[tracks.size()/2-1]->vz() + tracks[tracks.size()/2]->vz())/2;
  else
    vz =  tracks[tracks.size()/2  ]->vz();

  std::cerr << " [VertexProducer] median = " << vz << " cm" << std::endl;

  // Store
  reco::Vertex::Error err;
  err(2,2) = 1e-4*1e-4; // guess 10 um
  reco::Vertex        ver(reco::Vertex::Point(0,0,vz), err, 0, 1, 1);
  
  std::auto_ptr<reco::VertexCollection> vertices(new reco::VertexCollection);
  vertices->push_back(ver);
  ev.put(vertices);
}


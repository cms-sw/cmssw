#include "RecoPixelVertexing/PixelVertexFinding/interface/PixelVertexProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/DivisiveVertexFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <memory>
#include <string>
#include <cmath>

PixelVertexProducer::PixelVertexProducer(const edm::ParameterSet& conf) 
  : conf_(conf), verbose_(0), dvf_(0), ptMin_(1.0)
{
  // Register my product
  produces<reco::VertexCollection>();

  // Setup shop
  verbose_           = conf.getParameter<int>("Verbosity"); // 0 silent, 1 chatty, 2 loud
  std::string finder = conf.getParameter<std::string>("Finder"); // DivisiveVertexFinder
  bool useError      = conf.getParameter<bool>("UseError"); // true
  bool wtAverage     = conf.getParameter<bool>("WtAverage"); // true
  double zOffset     = conf.getParameter<double>("ZOffset"); // 5.0 sigma
  double zSeparation = conf.getParameter<double>("ZSeparation"); // 0.05 cm
  int ntrkMin        = conf.getParameter<int>("NTrkMin"); // 5
  // Tracking requirements before sending a track to be considered for vtx
  ptMin_ = conf_.getParameter<double>("PtMin"); // 1.0 GeV

  if (finder == "DivisiveVertexFinder") {
    if (verbose_ > 0) edm::LogInfo("PixelVertexProducer") << ": Using the DivisiveVertexFinder\n";
    dvf_ = new DivisiveVertexFinder(zOffset, ntrkMin, useError, zSeparation, wtAverage, verbose_);
  }
  else { // Finder not supported, or you made a mistake in your request
    // throw an exception once I figure out how CMSSW does this
  }
}


PixelVertexProducer::~PixelVertexProducer() {
  delete dvf_;
}

void PixelVertexProducer::produce(edm::Event& e, const edm::EventSetup& es) {

  // First fish the pixel tracks out of the event
  edm::Handle<reco::TrackCollection> trackCollection;
  std::string trackCollName = conf_.getParameter<std::string>("TrackCollection");
  e.getByLabel(trackCollName,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());
  if (verbose_ > 0) edm::LogInfo("PixelVertexProducer") << ": Found " << tracks.size() << " tracks in TrackCollection called " << trackCollName << "\n";
  

  // Second, make a collection of pointers to the tracks we want for the vertex finder
  reco::TrackRefVector trks;
  for (unsigned int i=0; i<tracks.size(); i++) {
    if (tracks[i].pt() > ptMin_)     
      trks.push_back( reco::TrackRef(trackCollection, i) );
  }
  if (verbose_ > 0) edm::LogInfo("PixelVertexProducer") << ": Selected " << trks.size() << " of these tracks for vertexing\n";

  // Third, ship these tracks off to be vertexed
  std::auto_ptr<reco::VertexCollection> vertexes(new reco::VertexCollection);
  bool ok;
  if (conf_.getParameter<bool>("Method2")) {
    ok = dvf_->findVertexesAlt(trks,       // input
			     *vertexes); // output
    if (verbose_ > 0) edm::LogInfo("PixelVertexProducer") << "Method2 returned status of " << ok;
  }
  else {
    ok = dvf_->findVertexes(trks,       // input
			    *vertexes); // output
    if (verbose_ > 0) edm::LogInfo("PixelVertexProducer") << "Method1 returned status of " << ok;
  }

  if (verbose_ > 0) {
    edm::LogInfo("PixelVertexProducer") << ": Found " << vertexes->size() << " vertexes\n";
    for (unsigned int i=0; i<vertexes->size(); ++i) {
      edm::LogInfo("PixelVertexProducer") << "Vertex number " << i << " has " << (*vertexes)[i].tracksSize() << " tracks with a position of " << (*vertexes)[i].z() << " +- " << std::sqrt( (*vertexes)[i].error(2,2) );
    }
  }

  // Finally, put them in the event if things look OK
  if (ok) e.put(vertexes);
}

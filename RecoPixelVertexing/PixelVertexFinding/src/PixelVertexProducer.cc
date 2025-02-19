#include "RecoPixelVertexing/PixelVertexFinding/interface/PixelVertexProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/DivisiveVertexFinder.h"
#include "FWCore/Utilities/interface/InputTag.h"
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
  int ntrkMin        = conf.getParameter<int>("NTrkMin"); // 3
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
  edm::InputTag trackCollName = conf_.getParameter<edm::InputTag>("TrackCollection");
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

  edm::Handle<reco::BeamSpot> bsHandle;
  edm::InputTag bsName = conf_.getParameter<edm::InputTag>("beamSpot");
  e.getByLabel(bsName,bsHandle);
  math::XYZPoint myPoint(0.,0.,0.);
  if (bsHandle.isValid()) myPoint = math::XYZPoint(bsHandle->x0(),bsHandle->y0(), 0. ); //FIXME: fix last coordinate with vertex.z() at same time

  // Third, ship these tracks off to be vertexed
  std::auto_ptr<reco::VertexCollection> vertexes(new reco::VertexCollection);
  bool ok;
  if (conf_.getParameter<bool>("Method2")) {
    ok = dvf_->findVertexesAlt(trks,       // input
			     *vertexes,myPoint); // output
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
      edm::LogInfo("PixelVertexProducer") << "Vertex number " << i << " has " << (*vertexes)[i].tracksSize() << " tracks with a position of " << (*vertexes)[i].z() << " +- " << std::sqrt( (*vertexes)[i].covariance(2,2) );
    }
  }


  if(bsHandle.isValid())
    {
      const reco::BeamSpot & bs = *bsHandle;
      
      for (unsigned int i=0; i<vertexes->size(); ++i) {
	double z=(*vertexes)[i].z();
	double x=bs.x0()+bs.dxdz()*(z-bs.z0());
	double y=bs.y0()+bs.dydz()*(z-bs.z0()); 
	reco::Vertex v( reco::Vertex::Point(x,y,z), (*vertexes)[i].error(),(*vertexes)[i].chi2() , (*vertexes)[i].ndof() , (*vertexes)[i].tracksSize());
	//Copy also the tracks 
	for (std::vector<reco::TrackBaseRef >::const_iterator it = (*vertexes)[i].tracks_begin();
	     it !=(*vertexes)[i].tracks_end(); it++) {
	  v.add( *it );
	}
	(*vertexes)[i]=v;
	
      }
    }
  else
    {
      edm::LogWarning("PixelVertexProducer") << "No beamspot found. Using returning vertexes with (0,0,Z) ";
    } 
  
  if(!vertexes->size() && bsHandle.isValid()){
    
    const reco::BeamSpot & bs = *bsHandle;
      
      GlobalError bse(bs.rotatedCovariance3D());
      if ( (bse.cxx() <= 0.) ||
	   (bse.cyy() <= 0.) ||
	   (bse.czz() <= 0.) ) {
	AlgebraicSymMatrix33 we;
	we(0,0)=10000;
	we(1,1)=10000;
	we(2,2)=10000;
	vertexes->push_back(reco::Vertex(bs.position(), we,0.,0.,0));
	
	edm::LogInfo("PixelVertexProducer") <<"No vertices found. Beamspot with invalid errors " << bse.matrix() << std::endl
					       << "Will put Vertex derived from dummy-fake BeamSpot into Event.\n"
					       << (*vertexes)[0].x() << "\n"
					       << (*vertexes)[0].y() << "\n"
					       << (*vertexes)[0].z() << "\n";
      } else {
	vertexes->push_back(reco::Vertex(bs.position(),
					 bs.rotatedCovariance3D(),0.,0.,0));
	
	edm::LogInfo("PixelVertexProducer") << "No vertices found. Will put Vertex derived from BeamSpot into Event:\n"
					       << (*vertexes)[0].x() << "\n"
					       << (*vertexes)[0].y() << "\n"
					       << (*vertexes)[0].z() << "\n";
      }
  }
      
  else if(!vertexes->size() && !bsHandle.isValid())
    {
      edm::LogWarning("PixelVertexProducer") << "No beamspot and no vertex found. No vertex returned.";
    }
  
  e.put(vertexes);
  
}



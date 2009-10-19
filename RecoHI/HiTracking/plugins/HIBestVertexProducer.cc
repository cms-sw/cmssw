#include "RecoHI/HiTracking/interface/HIBestVertexProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <iostream>
using namespace std;
using namespace edm;

/*****************************************************************************/
HIBestVertexProducer::HIBestVertexProducer
(const edm::ParameterSet& ps) : theConfig(ps)
{
  produces<reco::VertexCollection>();
}


/*****************************************************************************/
HIBestVertexProducer::~HIBestVertexProducer()
{ 
}

/*****************************************************************************/
void HIBestVertexProducer::beginJob
(const edm::EventSetup& es)
{
}

/*****************************************************************************/
void HIBestVertexProducer::produce
(edm::Event& ev, const edm::EventSetup& es)
{
  
  // New vertex collection
  std::auto_ptr<reco::VertexCollection> newVertex(new reco::VertexCollection);

  // Get reconstructed vertex collections ---------------------------
  math::XYZPoint vtxPoint(0.0,0.0,0.0);
  double vzErr =0.0;

  // Get precise adaptive vertex 
  edm::Handle<reco::VertexCollection> vc1;
  ev.getByLabel("hiBestAdaptiveVertex", vc1);
  const reco::VertexCollection * vertices1 = vc1.product();

  if(vertices1->size()>0) {
    vtxPoint = vertices1->begin()->position();
    vzErr = vertices1->begin()->zError();
    LogInfo("HeavyIonVertexing") << "Selected adaptive vertex:"
				 << "\n   vz = " << vtxPoint.Z()  
				 << "\n   vzErr = " << vzErr;
  } else {
    LogError("HeavyIonVertexing") << "No vertex found in collection '" 
                                  //<< vertexCollection1_ << "'";
				  << "hiSelectedVertex" << "'";
  }

  // Get fast median vertex
  edm::Handle<reco::VertexCollection> vc2;
  ev.getByLabel("hiPixelMedianVertex", vc2);
  const reco::VertexCollection * vertices2 = vc2.product();

  if(vertices2->size()>0) {
    vtxPoint = vertices2->begin()->position();
    vzErr = vertices2->begin()->zError();
    LogInfo("HeavyIonVertexing") << "Median vertex:"
				 << "\n   vz = " << vtxPoint.Z()  
				 << "\n   vzErr = " << vzErr;
  } else {
    LogError("HeavyIonVertexing") << "No vertex found in collection '" 
                                  //<< vertexCollection2_ << "'";
				  << "hiPixelMedianVertex" << "'";
  }

  // Get beamspot -------------------------------------------

  math::XYZPoint bsPoint(0.0,0.0,0.0);
  double bsWidth = 0.0;

  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  ev.getByLabel("offlineBeamSpot", beamSpotHandle);
  	
  if ( beamSpotHandle.isValid() ) {
    beamSpot = *beamSpotHandle;
    bsPoint = beamSpot.position();
    bsWidth = sqrt(beamSpot.BeamWidthX()*beamSpot.BeamWidthY());
    LogInfo("HeavyIonVertexing") << "Beamspot (x,y,z) = (" << bsPoint.X() 
				 << "," << bsPoint.Y() << "," << bsPoint.Z() 
				 << ")" << "\n   width = " << bsWidth;
  } else {
    edm::LogError("HeavyIonVertexing") << "No beam spot available from '" 
                                       //<< beamSpotLabel_ << "'\n";
				       << "offlineBeamSpot" << "'\n";
  }
  
  // Set vertex position and error ---------------------------
  reco::Vertex::Error err;
  err(2,2) = 0.1 * 0.1;
  reco::Vertex ver(reco::Vertex::Point(0,0,0),err, 0, 1, 1);
  newVertex->push_back(ver);
  
  ev.put(newVertex);

}


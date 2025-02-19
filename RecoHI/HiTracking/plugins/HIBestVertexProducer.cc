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
(const edm::ParameterSet& ps) : theConfig(ps),
  theBeamSpotTag(ps.getParameter<edm::InputTag>("beamSpotLabel")),
  theMedianVertexCollection(ps.getParameter<edm::InputTag>("medianVertexCollection")),
  theAdaptiveVertexCollection(ps.getParameter<edm::InputTag>("adaptiveVertexCollection"))
{
  produces<reco::VertexCollection>();
}


/*****************************************************************************/
HIBestVertexProducer::~HIBestVertexProducer()
{ 
}

/*****************************************************************************/
void HIBestVertexProducer::beginJob()
{
}

/*****************************************************************************/
void HIBestVertexProducer::produce
(edm::Event& ev, const edm::EventSetup& es)
{
  
  // 1. use best adaptive vertex preferentially
  // 2. use median vertex in case where adaptive algorithm failed
  // 3. use beamspot if netither vertexing method succeeds

  // New vertex collection
  std::auto_ptr<reco::VertexCollection> newVertexCollection(new reco::VertexCollection);

  //** Get precise adaptive vertex **/
  edm::Handle<reco::VertexCollection> vc1;
  ev.getByLabel(theAdaptiveVertexCollection, vc1);
  const reco::VertexCollection *vertices1 = vc1.product();

  if(vertices1->size()==0)
    LogError("HeavyIonVertexing") << "adaptive vertex collection is empty!" << endl;

  if(vertices1->begin()->zError()<3) { 
  
    reco::VertexCollection::const_iterator vertex1 = vertices1->begin();
    newVertexCollection->push_back(*vertex1);

    LogInfo("HeavyIonVertexing") << "adaptive vertex:\n vz = (" 
				 << vertex1->x() << ", " << vertex1->y() << ", " << vertex1->z() << ")" 
				 << "\n error = ("
				 << vertex1->xError() << ", " << vertex1->yError() << ", " 
				 << vertex1->zError() << ")" << endl;
  } else {
    
    //** Get fast median vertex **/
    edm::Handle<reco::VertexCollection> vc2;
    ev.getByLabel(theMedianVertexCollection, vc2);
    const reco::VertexCollection * vertices2 = vc2.product();
    
    //** Get beam spot position and error **/
    reco::BeamSpot beamSpot;
    edm::Handle<reco::BeamSpot> beamSpotHandle;
    ev.getByLabel(theBeamSpotTag, beamSpotHandle);

    if( beamSpotHandle.isValid() ) 
      beamSpot = *beamSpotHandle;
    else
      LogError("HeavyIonVertexing") << "no beamspot with name: '" << theBeamSpotTag << "'" << endl;

    if(vertices2->size() > 0) { 
      
      reco::VertexCollection::const_iterator vertex2 = vertices2->begin();
      reco::Vertex::Error err;
      err(0,0)=pow(beamSpot.BeamWidthX(),2);
      err(1,1)=pow(beamSpot.BeamWidthY(),2);
      err(2,2)=pow(vertex2->zError(),2);
      reco::Vertex newVertex(reco::Vertex::Point(beamSpot.x0(),beamSpot.y0(),vertex2->z()),
			     err, 0, 1, 1);
      newVertexCollection->push_back(newVertex);  

      LogInfo("HeavyIonVertexing") << "median vertex + beamspot: \n position = (" 
				   << newVertex.x() << ", " << newVertex.y() << ", " << newVertex.z() << ")" 
				   << "\n error = ("
				   << newVertex.xError() << ", " << newVertex.yError() << ", " 
				   << newVertex.zError() << ")" << endl;
      
    } else { 
      
      reco::Vertex::Error err;
      err(0,0)=pow(beamSpot.BeamWidthX(),2);
      err(1,1)=pow(beamSpot.BeamWidthY(),2);
      err(2,2)=pow(beamSpot.sigmaZ(),2);
      reco::Vertex newVertex(beamSpot.position(), 
			     err, 0, 0, 1);
      newVertexCollection->push_back(newVertex);  

      LogInfo("HeavyIonVertexing") << "beam spot: \n position = (" 
				   << newVertex.x() << ", " << newVertex.y() << ", " << newVertex.z() << ")" 
				   << "\n error = ("
				   << newVertex.xError() << ", " << newVertex.yError() << ", " 
				   << newVertex.zError() << ")" << endl;
      
    }
    
  }
  
  // put new vertex collection into event
  ev.put(newVertexCollection);
  
}


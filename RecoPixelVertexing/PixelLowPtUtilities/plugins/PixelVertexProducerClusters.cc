#include "PixelVertexProducerClusters.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

#include "TROOT.h"
#include "TH1F.h"
#include "TF1.h"

class VertexHit
{
  public:
    float z;
    float r;
    int w;
};

/*****************************************************************************/
PixelVertexProducerClusters::PixelVertexProducerClusters
  (const edm::ParameterSet& ps) : theConfig(ps)
{
  // Product
  produces<reco::VertexCollection>();
}


/*****************************************************************************/
PixelVertexProducerClusters::~PixelVertexProducerClusters()
{ 
}

/*****************************************************************************/
void PixelVertexProducerClusters::beginJob
  (const edm::EventSetup& es)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  theTracker = trackerHandle.product();
}

/*****************************************************************************/
int PixelVertexProducerClusters::getContainedHits
  (vector<VertexHit> hits, float z0, float & chi)
{
  int n = 0;
  chi = 0.;

  for(vector<VertexHit>::const_iterator hit = hits.begin();
                                        hit!= hits.end(); hit++)
  {
    // Predicted cluster width in y direction
    float p = 2 * fabs(hit->z - z0)/hit->r + 0.5; // FIXME

    if(fabs(p - hit->w) <= 1.)
    { 
      chi += fabs(p - hit->w);
      n++;
    }
  }

  return n;
}

/*****************************************************************************/
void PixelVertexProducerClusters::produce
  (edm::Event& ev, const edm::EventSetup& es)
{
/*
  edm::Handle<edm::SimVertexContainer> simVertices;
  ev.getByType(                        simVertices);

  float zsim = 0.;
  if((*(simVertices.product())).size() > 0)
    zsim = (*(simVertices.product())).at(0).position().z();
*/

  // Get pixel hit collections
  edm::Handle<SiPixelRecHitCollection> pixelColl;
  ev.getByLabel("siPixelRecHits",      pixelColl);

  const SiPixelRecHitCollection* thePixelHits = pixelColl.product();

  std::auto_ptr<reco::VertexCollection> vertices(new reco::VertexCollection);

  if(thePixelHits->size() > 0)
  {
    vector<VertexHit> hits;

    for(SiPixelRecHitCollection::DataContainer::const_iterator recHit = thePixelHits->data().begin(), recHitEnd = thePixelHits->data().end();
           recHit != recHitEnd; ++recHit)
    {
      if(recHit->isValid())
      if(!(recHit->isOnEdge() || recHit->hasBadPixels()))
      {
        DetId id = recHit->geographicalId();

        if(id.subdetId() == int(PixelSubdetector::PixelBarrel))  
        {
          PXBDetId pid(id);
//          int layer = pid.layer() - 1; // 0 - 2

          LocalPoint lpos = LocalPoint(recHit->localPosition().x(),
                                       recHit->localPosition().y(),
                                       recHit->localPosition().z());

          GlobalPoint gpos = theTracker->idToDet(id)->toGlobal(lpos);

/*
          file << " " << layer
               << " " << gpos.z()
               << " " << gpos.perp()
               << " " << recHit->cluster()->sizeY()
               << endl;
*/

          VertexHit hit;
          hit.z = gpos.z(); 
          hit.r = gpos.perp(); 
          hit.w = recHit->cluster()->sizeY();

          hits.push_back(hit);
        }
      }
    }

    int nhits; int nhits_max = 0;
    float chi; float chi_max = 1e+9;

    float zest = 0;
 
    for(float z0 = -15.9; z0 <= 15.95; z0 += 0.1)
    {
      nhits = getContainedHits(hits, z0, chi);

      if(nhits > 0)
      {
        if(nhits >  nhits_max)
        { chi_max = 1e+9; nhits_max = nhits; }

        if(nhits >= nhits_max)
        if(chi < chi_max)
        { chi_max = chi; zest = z0; }
      }
    }

    LogTrace("MinBiasTracking")
      << "  [vertex position] estimated = " << zest 
      << " | pixel barrel hits = " << thePixelHits->size();

    reco::Vertex::Error err;
    err(2,2) = 0.6 * 0.6;
    reco::Vertex ver(reco::Vertex::Point(0,0,zest), err, 0, 1, 1);
    vertices->push_back(ver);
  }

  ev.put(vertices);
}


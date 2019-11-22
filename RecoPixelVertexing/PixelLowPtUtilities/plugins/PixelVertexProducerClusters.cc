#include "PixelVertexProducerClusters.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <vector>
#include <algorithm>

using namespace std;

namespace {
  class VertexHit {
  public:
    float z;
    float r;
    int w;
  };

  /*****************************************************************************/
  int getContainedHits(const vector<VertexHit>& hits, float z0, float& chi) {
    int n = 0;
    chi = 0.;

    for (vector<VertexHit>::const_iterator hit = hits.begin(); hit != hits.end(); hit++) {
      // Predicted cluster width in y direction
      float p = 2 * fabs(hit->z - z0) / hit->r + 0.5;  // FIXME

      if (fabs(p - hit->w) <= 1.) {
        chi += fabs(p - hit->w);
        n++;
      }
    }

    return n;
  }
}  // namespace

/*****************************************************************************/
PixelVertexProducerClusters::PixelVertexProducerClusters(const edm::ParameterSet& ps)
    : pixelToken_(consumes<SiPixelRecHitCollection>(edm::InputTag("siPixelRecHits"))) {
  // Product
  produces<reco::VertexCollection>();
}

/*****************************************************************************/
PixelVertexProducerClusters::~PixelVertexProducerClusters() {}

/*****************************************************************************/
void PixelVertexProducerClusters::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<TrackerTopologyRcd>().get(tTopo);

  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  const TrackerGeometry* theTracker = trackerHandle.product();

  // Get pixel hit collections
  edm::Handle<SiPixelRecHitCollection> pixelColl;
  ev.getByToken(pixelToken_, pixelColl);

  const SiPixelRecHitCollection* thePixelHits = pixelColl.product();

  auto vertices = std::make_unique<reco::VertexCollection>();

  if (!thePixelHits->empty()) {
    vector<VertexHit> hits;

    for (SiPixelRecHitCollection::DataContainer::const_iterator recHit = thePixelHits->data().begin(),
                                                                recHitEnd = thePixelHits->data().end();
         recHit != recHitEnd;
         ++recHit) {
      if (recHit->isValid()) {
        //      if(!(recHit->isOnEdge() || recHit->hasBadPixels()))
        DetId id = recHit->geographicalId();
        const PixelGeomDetUnit* pgdu = dynamic_cast<const PixelGeomDetUnit*>(theTracker->idToDetUnit(id));
        const PixelTopology* theTopol = (&(pgdu->specificTopology()));
        vector<SiPixelCluster::Pixel> pixels = recHit->cluster()->pixels();

        bool pixelOnEdge = false;
        for (vector<SiPixelCluster::Pixel>::const_iterator pixel = pixels.begin(); pixel != pixels.end(); pixel++) {
          int pos_x = (int)pixel->x;
          int pos_y = (int)pixel->y;

          if (theTopol->isItEdgePixelInX(pos_x) || theTopol->isItEdgePixelInY(pos_y)) {
            pixelOnEdge = true;
            break;
          }
        }

        if (!pixelOnEdge)
          if (id.subdetId() == int(PixelSubdetector::PixelBarrel)) {
            LocalPoint lpos =
                LocalPoint(recHit->localPosition().x(), recHit->localPosition().y(), recHit->localPosition().z());

            GlobalPoint gpos = theTracker->idToDet(id)->toGlobal(lpos);

            VertexHit hit;
            hit.z = gpos.z();
            hit.r = gpos.perp();
            hit.w = recHit->cluster()->sizeY();

            hits.push_back(hit);
          }
      }
    }

    int nhits;
    int nhits_max = 0;
    float chi;
    float chi_max = 1e+9;

    float zest = 0;

    for (float z0 = -15.9; z0 <= 15.95; z0 += 0.1) {
      nhits = getContainedHits(hits, z0, chi);

      if (nhits > 0) {
        if (nhits > nhits_max) {
          chi_max = 1e+9;
          nhits_max = nhits;
        }

        if (nhits >= nhits_max)
          if (chi < chi_max) {
            chi_max = chi;
            zest = z0;
          }
      }
    }

    LogTrace("MinBiasTracking") << "  [vertex position] estimated = " << zest
                                << " | pixel barrel hits = " << thePixelHits->size();

    reco::Vertex::Error err;
    err(2, 2) = 0.6 * 0.6;
    reco::Vertex ver(reco::Vertex::Point(0, 0, zest), err, 0, 1, 1);
    vertices->push_back(ver);
  }

  ev.put(std::move(vertices));
}

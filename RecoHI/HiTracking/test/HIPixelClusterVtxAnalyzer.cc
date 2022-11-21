#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

// ROOT includes
#include <TH1.h>

class HIPixelClusterVtxAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HIPixelClusterVtxAnalyzer(const edm::ParameterSet &ps);

private:
  struct VertexHit {
    float z;
    float r;
    float w;
  };

  virtual void analyze(const edm::Event &ev, const edm::EventSetup &es);
  int getContainedHits(const std::vector<VertexHit> &hits, double z0, double &chi);

  const edm::EDGetTokenT<SiPixelRecHitCollection> srcPixels_;  //pixel rec hits
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerToken_;

  const double minZ_;
  const double maxZ_;
  const double zStep_;
  const int maxHists_;

  edm::Service<TFileService> fs;
  int counter;
};

/*****************************************************************************/
HIPixelClusterVtxAnalyzer::HIPixelClusterVtxAnalyzer(const edm::ParameterSet &ps)
    : srcPixels_(consumes<SiPixelRecHitCollection>(ps.getParameter<edm::InputTag>("pixelRecHits"))),
      trackerToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      minZ_(ps.getParameter<double>("minZ")),
      maxZ_(ps.getParameter<double>("maxZ")),
      zStep_(ps.getParameter<double>("zStep")),
      maxHists_(ps.getParameter<int>("maxHists")),
      counter(0)

{
  // Constructor
  usesResource(TFileService::kSharedResource);
}

/*****************************************************************************/
void HIPixelClusterVtxAnalyzer::analyze(const edm::Event &ev, const edm::EventSetup &es) {
  if (counter > maxHists_)
    return;
  std::cout << "counter = " << counter << std::endl;
  counter++;

  edm::EventNumber_t evtnum = ev.id().event();
  TH1D *hClusterVtx = fs->make<TH1D>(Form("hClusterVtx_%llu", evtnum),
                                     "compatibility of pixel cluster length with vertex hypothesis; z [cm]",
                                     (int)((maxZ_ - minZ_) / zStep_),
                                     minZ_,
                                     maxZ_);

  // get pixel rechits
  edm::Handle<SiPixelRecHitCollection> hRecHits;
  try {
    ev.getByToken(srcPixels_, hRecHits);
  } catch (...) {
  }

  // get tracker geometry
  if (hRecHits.isValid()) {
    const auto &trackerHandle = es.getHandle(trackerToken_);
    const TrackerGeometry *tgeo = trackerHandle.product();
    const SiPixelRecHitCollection *hits = hRecHits.product();

    // loop over pixel rechits
    std::vector<VertexHit> vhits;
    for (auto const &hit : hits->data()) {
      if (!hit.isValid())
        continue;
      DetId id(hit.geographicalId());
      if (id.subdetId() != int(PixelSubdetector::PixelBarrel))
        continue;
      const PixelGeomDetUnit *pgdu = static_cast<const PixelGeomDetUnit *>(tgeo->idToDet(id));
      if (1) {
        const RectangularPixelTopology *pixTopo =
            static_cast<const RectangularPixelTopology *>(&(pgdu->specificTopology()));
        std::vector<SiPixelCluster::Pixel> pixels(hit.cluster()->pixels());
        bool pixelOnEdge = false;
        for (std::vector<SiPixelCluster::Pixel>::const_iterator pixel = pixels.begin(); pixel != pixels.end();
             ++pixel) {
          int pixelX = pixel->x;
          int pixelY = pixel->y;
          if (pixTopo->isItEdgePixelInX(pixelX) || pixTopo->isItEdgePixelInY(pixelY)) {
            pixelOnEdge = true;
            break;
          }
        }
        if (pixelOnEdge)
          continue;
      }

      LocalPoint lpos = LocalPoint(hit.localPosition().x(), hit.localPosition().y(), hit.localPosition().z());
      GlobalPoint gpos = pgdu->toGlobal(lpos);
      VertexHit vh;
      vh.z = gpos.z();
      vh.r = gpos.perp();
      vh.w = hit.cluster()->sizeY();
      vhits.push_back(vh);
    }

    // estimate z-position from cluster lengths
    double zest = 0.0;
    int nhits = 0, nhits_max = 0;
    double chi = 0, chi_max = 1e+9;
    for (double z0 = minZ_; z0 <= maxZ_; z0 += zStep_) {
      nhits = getContainedHits(vhits, z0, chi);
      hClusterVtx->Fill(z0, nhits);
      if (nhits == 0)
        continue;
      if (nhits > nhits_max) {
        chi_max = 1e+9;
        nhits_max = nhits;
      }
      if (nhits >= nhits_max && chi < chi_max) {
        chi_max = chi;
        zest = z0;
      }
    }

    LogTrace("MinBiasTracking") << "  [vertex position] estimated = " << zest
                                << " | pixel barrel hits = " << vhits.size();
  }
}

/*****************************************************************************/
int HIPixelClusterVtxAnalyzer::getContainedHits(const std::vector<VertexHit> &hits, double z0, double &chi) {
  // Calculate number of hits contained in v-shaped window in cluster y-width vs. z-position.
  int n = 0;
  chi = 0.;

  for (std::vector<VertexHit>::const_iterator hit = hits.begin(); hit != hits.end(); hit++) {
    double p = 2 * fabs(hit->z - z0) / hit->r + 0.5;  // FIXME
    if (TMath::Abs(p - hit->w) <= 1.) {
      chi += fabs(p - hit->w);
      n++;
    }
  }
  return n;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HIPixelClusterVtxAnalyzer);

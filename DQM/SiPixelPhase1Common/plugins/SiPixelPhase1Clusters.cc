// -*- C++ -*-
//
// Package:     SiPixelPhase1Clusters
// Class:       SiPixelPhase1Clusters
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace {

  class SiPixelPhase1Clusters final : public SiPixelPhase1Base {
    enum {
      CHARGE,
      BIGPIXELCHARGE,
      NOTBIGPIXELCHARGE,
      SIZE,
      SIZEX,
      SIZEY,
      NCLUSTERS,
      NCLUSTERSINCLUSIVE,
      EVENTRATE,
      POSITION_B,
      POSITION_F,
      POSITION_XZ,
      POSITION_YZ,
      SIZE_VS_ETA,
      READOUT_CHARGE,
      READOUT_NCLUSTERS,
      PIXEL_TO_STRIP_RATIO
    };

  public:
    explicit SiPixelPhase1Clusters(const edm::ParameterSet& conf);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelSrcToken_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> stripSrcToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  };

  SiPixelPhase1Clusters::SiPixelPhase1Clusters(const edm::ParameterSet& iConfig)
      : SiPixelPhase1Base(iConfig), trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()) {
    pixelSrcToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("pixelSrc"));

    stripSrcToken_ = consumes<edmNew::DetSetVector<SiStripCluster>>(iConfig.getParameter<edm::InputTag>("stripSrc"));
  }

  void SiPixelPhase1Clusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    if (!checktrigger(iEvent, iSetup, DCS))
      return;

    edm::Handle<edmNew::DetSetVector<SiPixelCluster>> inputPixel;
    iEvent.getByToken(pixelSrcToken_, inputPixel);
    if (!inputPixel.isValid())
      return;

    edm::Handle<edmNew::DetSetVector<SiStripCluster>> inputStrip;
    iEvent.getByToken(stripSrcToken_, inputStrip);
    if (inputStrip.isValid()) {
      if (!inputStrip.product()->data().empty()) {
        histo[PIXEL_TO_STRIP_RATIO].fill(
            (double)inputPixel.product()->data().size() / (double)inputStrip.product()->data().size(),
            DetId(0),
            &iEvent);
      }
    }

    bool hasClusters = false;

    const TrackerGeometry& tracker = iSetup.getData(trackerGeometryToken_);

    edmNew::DetSetVector<SiPixelCluster>::const_iterator it;
    for (it = inputPixel->begin(); it != inputPixel->end(); ++it) {
      auto id = DetId(it->detId());

      const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(tracker.idToDet(id));
      const PixelTopology& topol = theGeomDet->specificTopology();

      for (SiPixelCluster const& cluster : *it) {
        int row = cluster.x() - 0.5, col = cluster.y() - 0.5;
        const std::vector<SiPixelCluster::Pixel> pixelsVec = cluster.pixels();

        for (unsigned int i = 0; i < pixelsVec.size(); ++i) {
          float pixx = pixelsVec[i].x;  // index as float=iteger, row index
          float pixy = pixelsVec[i].y;  // same, col index

          bool bigInX = topol.isItBigPixelInX(int(pixx));
          bool bigInY = topol.isItBigPixelInY(int(pixy));
          float pixel_charge = pixelsVec[i].adc;

          if (bigInX == true || bigInY == true) {
            histo[BIGPIXELCHARGE].fill(pixel_charge, id, &iEvent, col, row);
          }

          else {
            histo[NOTBIGPIXELCHARGE].fill(pixel_charge, id, &iEvent, col, row);
          }
        }
        histo[READOUT_CHARGE].fill(double(cluster.charge()), id, &iEvent, col, row);
        histo[CHARGE].fill(double(cluster.charge()), id, &iEvent, col, row);
        histo[SIZE].fill(double(cluster.size()), id, &iEvent, col, row);
        histo[SIZEX].fill(double(cluster.sizeX()), id, &iEvent, col, row);
        histo[SIZEY].fill(double(cluster.sizeY()), id, &iEvent, col, row);
        histo[NCLUSTERS].fill(id, &iEvent, col, row);
        histo[NCLUSTERSINCLUSIVE].fill(id, &iEvent);
        hasClusters = true;
        if (cluster.size() > 1) {
          histo[READOUT_NCLUSTERS].fill(id, &iEvent);
        }

        LocalPoint clustlp = topol.localPosition(MeasurementPoint(cluster.x(), cluster.y()));
        GlobalPoint clustgp = theGeomDet->surface().toGlobal(clustlp);
        histo[POSITION_B].fill(clustgp.z(), clustgp.phi(), id, &iEvent);
        histo[POSITION_F].fill(clustgp.x(), clustgp.y(), id, &iEvent);
        histo[POSITION_XZ].fill(clustgp.x(), clustgp.z(), id, &iEvent);
        histo[POSITION_YZ].fill(clustgp.y(), clustgp.z(), id, &iEvent);
        histo[SIZE_VS_ETA].fill(clustgp.eta(), cluster.sizeY(), id, &iEvent);
      }
    }

    if (hasClusters)
      histo[EVENTRATE].fill(DetId(0), &iEvent);

    histo[NCLUSTERS].executePerEventHarvesting(&iEvent);
    histo[READOUT_NCLUSTERS].executePerEventHarvesting(&iEvent);
    histo[NCLUSTERSINCLUSIVE].executePerEventHarvesting(&iEvent);
  }

}  // namespace

DEFINE_FWK_MODULE(SiPixelPhase1Clusters);

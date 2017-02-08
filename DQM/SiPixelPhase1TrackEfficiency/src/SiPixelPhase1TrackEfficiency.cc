// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackEfficiency
// Class:       SiPixelPhase1TrackEfficiency
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1TrackEfficiency/interface/SiPixelPhase1TrackEfficiency.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


SiPixelPhase1TrackEfficiency::SiPixelPhase1TrackEfficiency(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig) 
{
  trackAssociationToken_ = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajectories"));
  vtxToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryvertices"));
}

void SiPixelPhase1TrackEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // get geometry
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  assert(tracker.isValid());

  // get primary vertex
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken( vtxToken_, vertices);
  if (!vertices.isValid() || vertices->size() == 0) return;
  // should be used for weird cuts
  //const auto primaryVertex = vertices->at(0); 

  // get the map
  edm::Handle<TrajTrackAssociationCollection> ttac;
  iEvent.getByToken( trackAssociationToken_, ttac);

  for (auto& item : *ttac) {
    auto trajectory_ref = item.key;
    reco::TrackRef track_ref = item.val;

    bool isBpixtrack = false, isFpixtrack = false;
    int nStripHits = 0;

    // first, look at the full track to see whether it is good
    for (auto& measurement : trajectory_ref->measurements()) {
      // check if things are all valid
      if (!measurement.updatedState().isValid()) continue;
      auto hit = measurement.recHit();
      if (!hit->isValid()) continue;

      DetId id = hit->geographicalId();
      uint32_t subdetid = (id.subdetId());

      // count strip hits
      if(subdetid==StripSubdetector::TIB) nStripHits++;
      if(subdetid==StripSubdetector::TOB) nStripHits++;
      if(subdetid==StripSubdetector::TID) nStripHits++;
      if(subdetid==StripSubdetector::TEC) nStripHits++;

      // check that we are in the pixel
      if (subdetid == PixelSubdetector::PixelBarrel) isBpixtrack = true;
      if (subdetid == PixelSubdetector::PixelEndcap) isFpixtrack = true;
    }

    if (!isBpixtrack && !isFpixtrack) continue;

    // then, look at each hit
    for (auto& measurement : trajectory_ref->measurements()) {
      if (!measurement.updatedState().isValid()) continue;
      auto hit = measurement.recHit();

      DetId id = hit->geographicalId();
      uint32_t subdetid = (id.subdetId());
      if (   subdetid != PixelSubdetector::PixelBarrel 
          && subdetid != PixelSubdetector::PixelEndcap) continue;

      bool isHitValid   = hit->getType()==TrackingRecHit::valid;
      bool isHitMissing = hit->getType()==TrackingRecHit::missing;

      const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
      int row = 0, col = 0;
      if (pixhit) {
        const PixelGeomDetUnit* geomdetunit = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(id) );
        const PixelTopology& topol = geomdetunit->specificTopology();
        LocalPoint const& lp = pixhit->localPositionFast();
        MeasurementPoint mp = topol.measurementPosition(lp);
        row = (int) mp.x();
        col = (int) mp.y();
      }

      if (isHitValid)   histo[VALID  ].fill(id, &iEvent, col, row);
      if (isHitMissing) histo[MISSING].fill(id, &iEvent, col, row);
    }
  }
}

DEFINE_FWK_MODULE(SiPixelPhase1TrackEfficiency);


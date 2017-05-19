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
  tracksToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
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

  if (!vertices.isValid()) return;
  histo[VERTICES].fill(vertices->size(),DetId(0),&iEvent);
  if (vertices->size() == 0) return;

  // should be used for weird cuts
  //const auto primaryVertex = vertices->at(0); 

  // get the map
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken( tracksToken_, tracks);
  if (!tracks.isValid()) return;

  for (auto const & track : *tracks) {

    bool isBpixtrack = false, isFpixtrack = false;
    int nStripHits = 0;

    // first, look at the full track to see whether it is good
    auto const & trajParams = track.extra()->trajParams();
    auto hb = track.recHitsBegin();
    for(unsigned int h=0;h<track.recHitsSize();h++){
      
      auto hit = *(hb+h);
      if(!hit->isValid()) continue;

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
    for(unsigned int h=0;h<track.recHitsSize();h++){
      auto hit = *(hb+h);

      DetId id = hit->geographicalId();
      uint32_t subdetid = (id.subdetId());
      if (   subdetid != PixelSubdetector::PixelBarrel 
          && subdetid != PixelSubdetector::PixelEndcap) continue;

      bool isHitValid   = hit->getType()==TrackingRecHit::valid;
      bool isHitMissing = hit->getType()==TrackingRecHit::missing;

      const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(hit);
      const PixelGeomDetUnit* geomdetunit = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(id) );
      const PixelTopology& topol = geomdetunit->specificTopology();
      LocalPoint lp;
      if (pixhit) {
        lp = pixhit->localPosition();
      } else {
        lp = trajParams[h].position();
      }

      MeasurementPoint mp = topol.measurementPosition(lp);
      int row = (int) mp.x();
      int col = (int) mp.y();

      if (isHitValid)   {
        histo[VALID].fill(id, &iEvent, col, row);
        histo[EFFICIENCY].fill(1, id, &iEvent, col, row);
      }
      if (isHitMissing) {
        histo[MISSING].fill(id, &iEvent, col, row);
        histo[EFFICIENCY].fill(0, id, &iEvent, col, row);
      }
    }
  }
histo[VALID  ].executePerEventHarvesting(&iEvent);
histo[MISSING].executePerEventHarvesting(&iEvent);
}

DEFINE_FWK_MODULE(SiPixelPhase1TrackEfficiency);


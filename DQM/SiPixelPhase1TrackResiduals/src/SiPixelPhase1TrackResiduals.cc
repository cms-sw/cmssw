// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackResiduals
// Class:       SiPixelPhase1TrackResiduals
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1TrackResiduals/interface/SiPixelPhase1TrackResiduals.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

SiPixelPhase1TrackResiduals::SiPixelPhase1TrackResiduals(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig),
  validator(iConfig, consumesCollector())
{
  offlinePrimaryVerticesToken_ = consumes<reco::VertexCollection>(std::string("offlinePrimaryVertices"));
  ApplyVertexCut_=iConfig.getParameter<bool>("VertexCut");
}

void SiPixelPhase1TrackResiduals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  assert(tracker.isValid());

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(offlinePrimaryVerticesToken_, vertices);

  if (ApplyVertexCut_ && (!vertices.isValid() || vertices->size() == 0)) return;
  
  reco::Vertex primaryVertex; 
  if (ApplyVertexCut_) primaryVertex = vertices->at(0); 

  std::vector<TrackerValidationVariables::AVTrackStruct> vtracks;

  validator.fillTrackQuantities(iEvent, iSetup, 
    // tell the validator to only look at good tracks
    [&](const reco::Track& track) -> bool { 
	return (!ApplyVertexCut_ || (track.pt() > 0.75
	&& std::abs( track.dxy(primaryVertex.position()) ) < 5*track.dxyError())) ;
    }, vtracks);

  for (auto& track : vtracks) {
    for (auto& it : track.hits) {
      auto id = DetId(it.rawDetId);
      auto isPixel = id.subdetId() == 1 || id.subdetId() == 2;
      if (!isPixel) continue; 

      //TO BE UPDATED WITH VINCENZO STUFF
      const PixelGeomDetUnit* geomdetunit = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(id) );
      const PixelTopology& topol = geomdetunit->specificTopology();

      float lpx=it.localX;
      float lpy=it.localY;
      LocalPoint lp(lpx,lpy);

      MeasurementPoint mp = topol.measurementPosition(lp);
      int row = (int) mp.x();
      int col = (int) mp.y();

      histo[RESIDUAL_X].fill(it.resXprime, id, &iEvent, col, row);
      histo[RESIDUAL_Y].fill(it.resYprime, id, &iEvent, col, row);
    }
  }

}

DEFINE_FWK_MODULE(SiPixelPhase1TrackResiduals);


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


SiPixelPhase1TrackResiduals::SiPixelPhase1TrackResiduals(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig),
  validator(iConfig, consumesCollector())
{
  offlinePrimaryVerticesToken_ = consumes<reco::VertexCollection>(std::string("offlinePrimaryVertices"));
}

void SiPixelPhase1TrackResiduals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(offlinePrimaryVerticesToken_, vertices);
  if (!vertices.isValid() || vertices->size() == 0) return;
  const auto primaryVertex = vertices->at(0); 

  std::vector<TrackerValidationVariables::AVTrackStruct> vtracks;
  validator.fillTrackQuantities(iEvent, iSetup, 
    // tell the validator to only look at good tracks
    [&](const reco::Track& track) -> bool { 
      return track.pt() > 0.75
          && std::abs( track.dxy(primaryVertex.position()) ) < 5*track.dxyError();
    }, vtracks);

  for (auto& track : vtracks) {
    for (auto& it : track.hits) {
      auto id = DetId(it.rawDetId);
      auto isPixel = id.subdetId() == 1 || id.subdetId() == 2;
      if (!isPixel) continue; 

      histo[RESIDUAL_X].fill(it.resXprime, id, &iEvent);
      histo[RESIDUAL_Y].fill(it.resYprime, id, &iEvent);
    }
  }

}

DEFINE_FWK_MODULE(SiPixelPhase1TrackResiduals);


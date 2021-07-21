// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackResiduals
// Class:       SiPixelPhase1TrackResiduals
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

namespace {

  class SiPixelPhase1TrackResiduals final : public SiPixelPhase1Base {
    enum {
      RESIDUAL_X,
      RESIDUAL_Y,
      RESONEDGE_X,
      RESONEDGE_Y,
      RESOTHERBAD_X,
      RESOTHERBAD_Y,
      NormRes_X,
      NormRes_Y,
      DRNR_X,
      DRNR_Y
    };

  public:
    explicit SiPixelPhase1TrackResiduals(const edm::ParameterSet& conf);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    TrackerValidationVariables validator;
    edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;

    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;

    bool applyVertexCut_;
  };

  SiPixelPhase1TrackResiduals::SiPixelPhase1TrackResiduals(const edm::ParameterSet& iConfig)
      : SiPixelPhase1Base(iConfig), validator(iConfig, consumesCollector()) {
    applyVertexCut_ = iConfig.getUntrackedParameter<bool>("VertexCut", true);

    offlinePrimaryVerticesToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"));

    trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  }

  void SiPixelPhase1TrackResiduals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    if (!checktrigger(iEvent, iSetup, DCS))
      return;

    edm::ESHandle<TrackerGeometry> tracker = iSetup.getHandle(trackerGeomToken_);
    assert(tracker.isValid());

    edm::Handle<reco::VertexCollection> vertices;
    if (applyVertexCut_) {
      iEvent.getByToken(offlinePrimaryVerticesToken_, vertices);
      if (!vertices.isValid() || vertices->empty())
        return;
    }

    std::vector<TrackerValidationVariables::AVTrackStruct> vtracks;

    validator.fillTrackQuantities(
        iEvent,
        iSetup,
        // tell the validator to only look at good tracks
        [&](const reco::Track& track) -> bool {
          return (!applyVertexCut_ ||
                  (track.pt() > 0.75 && std::abs(track.dxy(vertices->at(0).position())) < 5 * track.dxyError()));
        },
        vtracks);

    for (auto& track : vtracks) {
      for (auto& it : track.hits) {
        auto id = DetId(it.rawDetId);
        auto isPixel = id.subdetId() == 1 || id.subdetId() == 2;
        if (!isPixel)
          continue;

        histo[RESIDUAL_X].fill(it.resXprime, id, &iEvent);
        histo[RESIDUAL_Y].fill(it.resYprime, id, &iEvent);

        if (it.resXprimeErr != 0 && it.resYprimeErr != 0) {
          histo[NormRes_X].fill(it.resXprime / it.resXprimeErr, id, &iEvent);
          histo[NormRes_Y].fill(it.resYprime / it.resYprimeErr, id, &iEvent);
          histo[DRNR_X].fill(it.resXprime / it.resXprimeErr, id, &iEvent);
          histo[DRNR_Y].fill(it.resYprime / it.resYprimeErr, id, &iEvent);
        }

        if (it.isOnEdgePixel) {
          histo[RESONEDGE_X].fill(it.resXprime, id, &iEvent);
          histo[RESONEDGE_Y].fill(it.resYprime, id, &iEvent);
        }

        if (it.isOtherBadPixel) {
          histo[RESOTHERBAD_X].fill(it.resXprime, id, &iEvent);
          histo[RESOTHERBAD_Y].fill(it.resYprime, id, &iEvent);
        }
      }
    }
  }

}  // namespace

DEFINE_FWK_MODULE(SiPixelPhase1TrackResiduals);

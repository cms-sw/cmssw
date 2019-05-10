
/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 */

#include "DQM/DTMonitorModule/interface/DTCalibValidationFromMuons.h"

// Framework
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Geometry
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// RecHit
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include <iterator>

using namespace edm;
using namespace std;

DTCalibValidationFromMuons::DTCalibValidationFromMuons(const ParameterSet &pset) {
  parameters = pset;

  // the name of the 4D segments
  segment4DToken_ =
      consumes<DTRecSegment4DCollection>(edm::InputTag(parameters.getUntrackedParameter<string>("segment4DLabel")));
  // muon collection for matching 4D segments to muons
  muonToken_ = consumes<reco::MuonCollection>(edm::InputTag(parameters.getUntrackedParameter<string>("muonLabel")));
  // the counter of segments not used to compute residuals
  wrongSegment = 0;
  // the counter of segments used to compute residuals
  rightSegment = 0;

  nevent = 0;
}

DTCalibValidationFromMuons::~DTCalibValidationFromMuons() {
  // FR the following was previously in the endJob

  LogVerbatim("DTCalibValidationFromMuons") << "Segments used to compute residuals: " << rightSegment;
  LogVerbatim("DTCalibValidationFromMuons") << "Segments not used to compute residuals: " << wrongSegment;
}

void DTCalibValidationFromMuons::dqmBeginRun(const edm::Run &run, const edm::EventSetup &setup) {
  // get the geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);
}

void DTCalibValidationFromMuons::analyze(const edm::Event &event, const edm::EventSetup &setup) {
  ++nevent;
  LogTrace("DTCalibValidationFromMuons") << "[DTCalibValidationFromMuons] Analyze #Run: " << event.id().run()
                                         << " #Event: " << nevent;

  // RecHit mapping at Step 3 ---------------------------------
  LogTrace("DTCalibValidationFromMuons") << "  -- DTRecHit S3: begin analysis:";
  // Get the 4D rechits from the event
  Handle<reco::MuonCollection> muonH;
  event.getByToken(muonToken_, muonH);
  const vector<reco::Muon> *muons = muonH.product();

  // Get the 4D rechits from the event
  Handle<DTRecSegment4DCollection> segment4Ds;
  event.getByToken(segment4DToken_, segment4Ds);

  vector<const DTRecSegment4D *> selectedSegment4Ds;

  for (auto &imuon : *muons) {
    for (const auto &ch : imuon.matches()) {
      DetId chId(ch.id.rawId());
      if (chId.det() != DetId::Muon)
        continue;
      if (chId.subdetId() != MuonSubdetId::DT)
        continue;
      if (imuon.pt() < 15)
        continue;
      if (!imuon.isGlobalMuon())
        continue;

      int nsegs = ch.segmentMatches.size();
      if (!nsegs)
        continue;

      // get the DT segments that were used to construct the muon
      DTChamberId matchId = ch.id();
      DTRecSegment4DCollection::range segs = segment4Ds->get(matchId);
      for (DTRecSegment4DCollection::const_iterator segment = segs.first; segment != segs.second; ++segment) {
        LocalPoint posHit = segment->localPosition();
        float dx = (posHit.x() ? posHit.x() - ch.x : 0);
        float dy = (posHit.y() ? posHit.y() - ch.y : 0);
        float dr = sqrt(dx * dx + dy * dy);
        if (dr < 5)
          selectedSegment4Ds.push_back(&(*segment));
      }
    }
  }

  // Loop over all 4D segments
  for (auto segment : selectedSegment4Ds) {
    LogTrace("DTCalibValidationFromMuons") << "Anlysis on recHit at step 3";
    compute(dtGeom.product(), *segment);
  }
}

// Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
float DTCalibValidationFromMuons::recHitDistFromWire(const DTRecHit1DPair &hitPair, const DTLayer *layer) {
  return fabs(hitPair.localPosition(DTEnums::Left).x() - hitPair.localPosition(DTEnums::Right).x()) / 2.;
}

// Compute the distance from wire (cm) of a hits in a DTRecHit1D
float DTCalibValidationFromMuons::recHitDistFromWire(const DTRecHit1D &recHit, const DTLayer *layer) {
  return fabs(recHit.localPosition().x() - layer->specificTopology().wirePosition(recHit.wireId().wire()));
}

// Compute the position (cm) of a hits in a DTRecHit1DPair
float DTCalibValidationFromMuons::recHitPosition(
    const DTRecHit1DPair &hitPair, const DTLayer *layer, const DTChamber *chamber, float segmentPos, int sl) {
  // Get the layer and the wire position
  GlobalPoint hitPosGlob_right = layer->toGlobal(hitPair.localPosition(DTEnums::Right));
  LocalPoint hitPosInChamber_right = chamber->toLocal(hitPosGlob_right);
  GlobalPoint hitPosGlob_left = layer->toGlobal(hitPair.localPosition(DTEnums::Left));
  LocalPoint hitPosInChamber_left = chamber->toLocal(hitPosGlob_left);

  float recHitPos = -1;
  if (sl != 2) {
    if (fabs(hitPosInChamber_left.x() - segmentPos) < fabs(hitPosInChamber_right.x() - segmentPos))
      recHitPos = hitPosInChamber_left.x();
    else
      recHitPos = hitPosInChamber_right.x();
  } else {
    if (fabs(hitPosInChamber_left.y() - segmentPos) < fabs(hitPosInChamber_right.y() - segmentPos))
      recHitPos = hitPosInChamber_left.y();
    else
      recHitPos = hitPosInChamber_right.y();
  }

  return recHitPos;
}

// Compute the position (cm) of a hits in a  DTRecHit1D
float DTCalibValidationFromMuons::recHitPosition(
    const DTRecHit1D &recHit, const DTLayer *layer, const DTChamber *chamber, float segmentPos, int sl) {
  // Get the layer and the wire position
  GlobalPoint recHitPosGlob = layer->toGlobal(recHit.localPosition());
  LocalPoint recHitPosInChamber = chamber->toLocal(recHitPosGlob);

  float recHitPos = -1;
  if (sl != 2)
    recHitPos = recHitPosInChamber.x();
  else
    recHitPos = recHitPosInChamber.y();

  return recHitPos;
}

// Compute the residuals
void DTCalibValidationFromMuons::compute(const DTGeometry *dtGeom, const DTRecSegment4D &segment) {
  bool computeResidual = true;

  // Get all 1D RecHits at step 3 within the 4D segment
  vector<DTRecHit1D> recHits1D_S3;

  // Get 1D RecHits at Step 3 and select only events with
  // >=7 hits in phi and 4 hits in theta (if any)
  const DTChamberRecSegment2D *phiSeg = segment.phiSegment();
  if (phiSeg) {
    vector<DTRecHit1D> phiRecHits = phiSeg->specificRecHits();
    if (phiRecHits.size() < 7) {
      LogTrace("DTCalibValidationFromMuons") << "[DTCalibValidationFromMuons] Phi segments has: " << phiRecHits.size()
                                             << " hits, skipping";  // FIXME: info output
      computeResidual = false;
    }
    copy(phiRecHits.begin(), phiRecHits.end(), back_inserter(recHits1D_S3));
  }
  if (!phiSeg) {
    LogTrace("DTCalibValidationFromMuons") << " [DTCalibValidationFromMuons] 4D segment has no phi segment! ";
    computeResidual = false;
  }

  if (segment.dimension() == 4) {
    const DTSLRecSegment2D *zSeg = segment.zSegment();
    if (zSeg) {
      vector<DTRecHit1D> zRecHits = zSeg->specificRecHits();
      if (zRecHits.size() != 4) {
        LogTrace("DTCalibValidationFromMuons") << "[DTCalibValidationFromMuons] Theta segments has: " << zRecHits.size()
                                               << " hits, skipping";  // FIXME: info output
        computeResidual = false;
      }
      copy(zRecHits.begin(), zRecHits.end(), back_inserter(recHits1D_S3));
    }
    if (!zSeg) {
      LogTrace("DTCalibValidationFromMuons") << " [DTCalibValidationFromMuons] 4D segment has not the z segment! ";
      computeResidual = false;
    }
  }

  if (!computeResidual)
    ++wrongSegment;

  if (computeResidual) {
    ++rightSegment;

    // Loop over 1D RecHit inside 4D segment
    for (vector<DTRecHit1D>::const_iterator recHit1D = recHits1D_S3.begin(); recHit1D != recHits1D_S3.end();
         ++recHit1D) {
      const DTWireId wireId = (*recHit1D).wireId();

      // Get the layer and the wire position
      const DTLayer *layer = dtGeom->layer(wireId);
      float wireX = layer->specificTopology().wirePosition(wireId.wire());

      // Extrapolate the segment to the z of the wire
      // Get wire position in chamber RF
      // (y and z must be those of the hit to be coherent in the transf. of RF
      // in case of rotations of the layer alignment)
      LocalPoint wirePosInLay(wireX, (*recHit1D).localPosition().y(), (*recHit1D).localPosition().z());
      GlobalPoint wirePosGlob = layer->toGlobal(wirePosInLay);
      const DTChamber *chamber = dtGeom->chamber((*recHit1D).wireId().layerId().chamberId());
      LocalPoint wirePosInChamber = chamber->toLocal(wirePosGlob);

      // Segment position at Wire z in chamber local frame
      LocalPoint segPosAtZWire = segment.localPosition() + segment.localDirection() * wirePosInChamber.z() /
                                                               cos(segment.localDirection().theta());

      // Compute the distance of the segment from the wire
      int sl = wireId.superlayer();
      float SegmDistance = -1;
      if (sl == 1 || sl == 3) {
        // RPhi SL
        SegmDistance = fabs(wirePosInChamber.x() - segPosAtZWire.x());
        LogTrace("DTCalibValidationFromMuons") << "SegmDistance: " << SegmDistance;
      } else if (sl == 2) {
        // RZ SL
        SegmDistance = fabs(segPosAtZWire.y() - wirePosInChamber.y());
        LogTrace("DTCalibValidationFromMuons") << "SegmDistance: " << SegmDistance;
      }

      if (SegmDistance > 2.1)
        LogTrace("DTCalibValidationFromMuons") << "  Warning: dist segment-wire: " << SegmDistance;

      // Compute the distance of the recHit from the wire
      float recHitWireDist = recHitDistFromWire(*recHit1D, layer);
      LogTrace("DTCalibValidationFromMuons") << "recHitWireDist: " << recHitWireDist;

      // Compute the residuals
      float residualOnDistance = recHitWireDist - SegmDistance;
      LogTrace("DTCalibValidationFromMuons") << "WireId: " << wireId << "  ResidualOnDistance: " << residualOnDistance;
      float residualOnPosition = -1;
      float recHitPos = -1;
      if (sl == 1 || sl == 3) {
        recHitPos = recHitPosition(*recHit1D, layer, chamber, segPosAtZWire.x(), sl);
        residualOnPosition = recHitPos - segPosAtZWire.x();
      } else {
        recHitPos = recHitPosition(*recHit1D, layer, chamber, segPosAtZWire.y(), sl);
        residualOnPosition = recHitPos - segPosAtZWire.y();
      }
      LogTrace("DTCalibValidationFromMuons") << "WireId: " << wireId << "  ResidualOnPosition: " << residualOnPosition;

      // Fill the histos
      if (sl == 1 || sl == 3)
        fillHistos(wireId.superlayerId(),
                   SegmDistance,
                   residualOnDistance,
                   (wirePosInChamber.x() - segPosAtZWire.x()),
                   residualOnPosition,
                   3);
      else
        fillHistos(wireId.superlayerId(),
                   SegmDistance,
                   residualOnDistance,
                   (wirePosInChamber.y() - segPosAtZWire.y()),
                   residualOnPosition,
                   3);
    }
  }
}

void DTCalibValidationFromMuons::bookHistograms(DQMStore::IBooker &ibooker,
                                                edm::Run const &iRun,
                                                edm::EventSetup const &iSetup) {
  // FR substitute the DQMStore instance by ibooker
  ibooker.setCurrentFolder("DT/DTCalibValidationFromMuons");

  DTSuperLayerId slId;

  // Loop over all the chambers
  vector<const DTChamber *>::const_iterator ch_it = dtGeom->chambers().begin();
  vector<const DTChamber *>::const_iterator ch_end = dtGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {
    vector<const DTSuperLayer *>::const_iterator sl_it = (*ch_it)->superLayers().begin();
    vector<const DTSuperLayer *>::const_iterator sl_end = (*ch_it)->superLayers().end();
    // Loop over the SLs
    for (; sl_it != sl_end; ++sl_it) {
      slId = (*sl_it)->id();

      // TODO! fix this is a leftover
      int firstStep = 3;
      // Loop over the 3 steps
      for (int step = firstStep; step <= 3; ++step) {
        LogTrace("DTCalibValidationFromMuons") << "   Booking histos for SL: " << slId;

        // Compose the chamber name
        stringstream wheel;
        wheel << slId.wheel();
        stringstream station;
        station << slId.station();
        stringstream sector;
        sector << slId.sector();
        stringstream superLayer;
        superLayer << slId.superlayer();
        // Define the step
        stringstream Step;
        Step << step;

        string slHistoName = "_STEP" + Step.str() + "_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() +
                             "_SL" + superLayer.str();

        ibooker.setCurrentFolder("DT/DTCalibValidationFromMuons/Wheel" + wheel.str() + "/Station" + station.str() +
                                 "/Sector" + sector.str());
        // Create the monitor elements
        vector<MonitorElement *> histos;
        // Note the order matters
        histos.push_back(ibooker.book1D(
            "hResDist" + slHistoName, "Residuals on the distance from wire (rec_hit - segm_extr) (cm)", 200, -0.4, 0.4));
        histos.push_back(ibooker.book2D("hResDistVsDist" + slHistoName,
                                        "Residuals on the distance (cm) from wire (rec_hit "
                                        "- segm_extr) vs distance  (cm)",
                                        100,
                                        0,
                                        2.5,
                                        200,
                                        -0.4,
                                        0.4));

        histosPerSL[make_pair(slId, step)] = histos;
      }
    }
  }
}

// Fill a set of histograms for a given SL
void DTCalibValidationFromMuons::fillHistos(
    DTSuperLayerId slId, float distance, float residualOnDistance, float position, float residualOnPosition, int step) {
  // FIXME: optimization of the number of searches
  vector<MonitorElement *> histos = histosPerSL[make_pair(slId, step)];
  histos[0]->Fill(residualOnDistance);
  histos[1]->Fill(distance, residualOnDistance);
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:

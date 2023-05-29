
/*
 *  See header file for a description of this class.
 *
 */

#include "DTResidualCalibration.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Geometry
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "CommonTools/Utils/interface/TH1AddDirectorySentry.h"
#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"
#include "CalibMuon/DTCalibration/interface/DTRecHitSegmentResidual.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

#include <algorithm>

DTResidualCalibration::DTResidualCalibration(const edm::ParameterSet& pset)
    : histRange_(pset.getParameter<double>("histogramRange")),
      segment4DToken_(consumes<DTRecSegment4DCollection>(pset.getParameter<edm::InputTag>("segment4DLabel"))),
      rootBaseDir_(pset.getUntrackedParameter<std::string>("rootBaseDir", "DT/Residuals")),
      detailedAnalysis_(pset.getUntrackedParameter<bool>("detailedAnalysis", false)),
      dtGeomToken_(esConsumes<edm::Transition::BeginRun>()) {
  edm::ConsumesCollector collector(consumesCollector());
  select_ = new DTSegmentSelector(pset, collector);

  LogDebug("Calibration") << "[DTResidualCalibration] Constructor called.";
  std::string rootFileName = pset.getUntrackedParameter<std::string>("rootFileName", "residuals.root");
  rootFile_ = new TFile(rootFileName.c_str(), "RECREATE");
  rootFile_->cd();

  segmok = 0;
  segmbad = 0;
  nevent = 0;
}

DTResidualCalibration::~DTResidualCalibration() {
  delete select_;
  edm::LogVerbatim("Calibration") << "[DTResidualCalibration] Destructor called.";
  edm::LogVerbatim("Calibration") << "[DTResidualCalibration] Analyzed events: " << nevent;
  edm::LogVerbatim("Calibration") << "[DTResidualCalibration] Good segments: " << segmok;
  edm::LogVerbatim("Calibration") << "[DTResidualCalibration] Bad segments: " << segmbad;
}

void DTResidualCalibration::beginJob() { TH1::SetDefaultSumw2(true); }

void DTResidualCalibration::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  // get the geometry
  edm::ESHandle<DTGeometry> dtGeomH;
  dtGeomH = setup.getHandle(dtGeomToken_);
  dtGeom_ = dtGeomH.product();

  // Loop over all the chambers
  if (histoMapTH1F_.empty()) {
    for (auto ch_it : dtGeom_->chambers()) {
      // Loop over the SLs
      for (auto sl_it : ch_it->superLayers()) {
        DTSuperLayerId slId = (sl_it)->id();
        bookHistos(slId);
        if (detailedAnalysis_) {
          for (auto layer_it : (sl_it)->layers()) {
            DTLayerId layerId = (layer_it)->id();
            bookHistos(layerId);
          }
        }
      }
    }
  }
}

void DTResidualCalibration::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  rootFile_->cd();
  ++nevent;

  // Get the 4D rechits from the event
  const edm::Handle<DTRecSegment4DCollection>& segments4D = event.getHandle(segment4DToken_);

  // Loop over segments by chamber
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for (chamberIdIt = segments4D->id_begin(); chamberIdIt != segments4D->id_end(); ++chamberIdIt) {
    const DTChamber* chamber = dtGeom_->chamber(*chamberIdIt);

    // Get the range for the corresponding ChamberId
    DTRecSegment4DCollection::range range = segments4D->get((*chamberIdIt));
    // Loop over the rechits of this DetUnit
    for (DTRecSegment4DCollection::const_iterator segment = range.first; segment != range.second; ++segment) {
      LogTrace("Calibration") << "Segment local pos (in chamber RF): " << (*segment).localPosition()
                              << "\nSegment global pos: " << chamber->toGlobal((*segment).localPosition());

      if (!(*select_)(*segment, event, setup)) {
        segmbad++;
        continue;
      }
      segmok++;

      // Get all 1D RecHits at step 3 within the 4D segment
      std::vector<DTRecHit1D> recHits1D_S3;

      if ((*segment).hasPhi()) {
        const DTChamberRecSegment2D* phiSeg = (*segment).phiSegment();
        const std::vector<DTRecHit1D>& phiRecHits = phiSeg->specificRecHits();
        std::copy(phiRecHits.begin(), phiRecHits.end(), back_inserter(recHits1D_S3));
      }

      if ((*segment).hasZed()) {
        const DTSLRecSegment2D* zSeg = (*segment).zSegment();
        const std::vector<DTRecHit1D>& zRecHits = zSeg->specificRecHits();
        std::copy(zRecHits.begin(), zRecHits.end(), back_inserter(recHits1D_S3));
      }

      // Loop over 1D RecHit inside 4D segment
      for (std::vector<DTRecHit1D>::const_iterator recHit1D = recHits1D_S3.begin(); recHit1D != recHits1D_S3.end();
           ++recHit1D) {
        const DTWireId wireId = recHit1D->wireId();

        float segmDistance = segmentToWireDistance(*recHit1D, *segment);
        if (segmDistance > 2.1)
          LogTrace("Calibration") << "WARNING: segment-wire distance: " << segmDistance;
        else
          LogTrace("Calibration") << "segment-wire distance: " << segmDistance;

        float residualOnDistance = DTRecHitSegmentResidual().compute(dtGeom_, *recHit1D, *segment);
        LogTrace("Calibration") << "Wire Id " << wireId << " residual on distance: " << residualOnDistance;

        fillHistos(wireId.superlayerId(), segmDistance, residualOnDistance);
        if (detailedAnalysis_)
          fillHistos(wireId.layerId(), segmDistance, residualOnDistance);
      }
    }
  }
}

float DTResidualCalibration::segmentToWireDistance(const DTRecHit1D& recHit1D, const DTRecSegment4D& segment) {
  // Get the layer and the wire position
  const DTWireId wireId = recHit1D.wireId();
  const DTLayer* layer = dtGeom_->layer(wireId);
  float wireX = layer->specificTopology().wirePosition(wireId.wire());

  // Extrapolate the segment to the z of the wire
  // Get wire position in chamber RF
  // (y and z must be those of the hit to be coherent in the transf. of RF in case of rotations of the layer alignment)
  LocalPoint wirePosInLay(wireX, recHit1D.localPosition().y(), recHit1D.localPosition().z());
  GlobalPoint wirePosGlob = layer->toGlobal(wirePosInLay);
  const DTChamber* chamber = dtGeom_->chamber(wireId.layerId().chamberId());
  LocalPoint wirePosInChamber = chamber->toLocal(wirePosGlob);

  // Segment position at Wire z in chamber local frame
  LocalPoint segPosAtZWire =
      segment.localPosition() + segment.localDirection() * wirePosInChamber.z() / cos(segment.localDirection().theta());

  // Compute the distance of the segment from the wire
  int sl = wireId.superlayer();
  float segmDistance = -1;
  if (sl == 1 || sl == 3)
    segmDistance = fabs(wirePosInChamber.x() - segPosAtZWire.x());
  else if (sl == 2)
    segmDistance = fabs(segPosAtZWire.y() - wirePosInChamber.y());

  return segmDistance;
}

void DTResidualCalibration::endJob() {
  LogDebug("Calibration") << "[DTResidualCalibration] Writing histos to file.";
  rootFile_->cd();
  rootFile_->Write();
  rootFile_->Close();

  /*std::map<DTSuperLayerId, TH1F* >::const_iterator itSlHistos = histoMapTH1F_.begin();
  std::map<DTSuperLayerId, TH1F* >::const_iterator itSlHistos_end = histoMapTH1F_.end(); 
  for(; itSlHistos != itSlHistos_end; ++itSlHistos){
     std::vector<TH1F*>::const_iterator itHistTH1F = (*itSlHistos).second.begin();
     std::vector<TH1F*>::const_iterator itHistTH1F_end = (*itSlHistos).second.end();
     for(; itHistTH1F != itHistTH1F_end; ++itHistTH1F) (*itHistTH1F)->Write();

     std::vector<TH2F*>::const_iterator itHistTH2F = histoMapTH2F_[(*itSlHistos).first].begin();
     std::vector<TH2F*>::const_iterator itHistTH2F_end = histoMapTH2F_[(*itSlHistos).first].end();
     for(; itHistTH2F != itHistTH2F_end; ++itHistTH2F) (*itHistTH2F)->Write();
  }*/
}

void DTResidualCalibration::bookHistos(DTSuperLayerId slId) {
  TH1AddDirectorySentry addDir;
  rootFile_->cd();

  LogDebug("Calibration") << "[DTResidualCalibration] Booking histos for SL: " << slId;

  // Compose the chamber name
  // Define the step
  int step = 3;

  std::string wheelStr = std::to_string(slId.wheel());
  std::string stationStr = std::to_string(slId.station());
  std::string sectorStr = std::to_string(slId.sector());

  std::string slHistoName = "_STEP" + std::to_string(step) + "_W" + wheelStr + "_St" + stationStr + "_Sec" + sectorStr +
                            "_SL" + std::to_string(slId.superlayer());

  LogDebug("Calibration") << "Accessing " << rootBaseDir_;
  TDirectory* baseDir = rootFile_->GetDirectory(rootBaseDir_.c_str());
  if (!baseDir)
    baseDir = rootFile_->mkdir(rootBaseDir_.c_str());
  LogDebug("Calibration") << "Accessing " << ("Wheel" + wheelStr);
  TDirectory* wheelDir = baseDir->GetDirectory(("Wheel" + wheelStr).c_str());
  if (!wheelDir)
    wheelDir = baseDir->mkdir(("Wheel" + wheelStr).c_str());
  LogDebug("Calibration") << "Accessing " << ("Station" + stationStr);
  TDirectory* stationDir = wheelDir->GetDirectory(("Station" + stationStr).c_str());
  if (!stationDir)
    stationDir = wheelDir->mkdir(("Station" + stationStr).c_str());
  LogDebug("Calibration") << "Accessing " << ("Sector" + sectorStr);
  TDirectory* sectorDir = stationDir->GetDirectory(("Sector" + sectorStr).c_str());
  if (!sectorDir)
    sectorDir = stationDir->mkdir(("Sector" + sectorStr).c_str());

  sectorDir->cd();

  // Create the monitor elements
  TH1F* histosTH1F = new TH1F(("hResDist" + slHistoName).c_str(),
                              "Residuals on the distance from wire (rec_hit - segm_extr) (cm)",
                              200,
                              -histRange_,
                              histRange_);
  TH2F* histosTH2F = new TH2F(("hResDistVsDist" + slHistoName).c_str(),
                              "Residuals on the dist. (cm) from wire (rec_hit - segm_extr) vs dist. (cm)",
                              100,
                              0,
                              2.5,
                              200,
                              -histRange_,
                              histRange_);
  histoMapTH1F_[slId] = histosTH1F;
  histoMapTH2F_[slId] = histosTH2F;
}

void DTResidualCalibration::bookHistos(DTLayerId layerId) {
  TH1AddDirectorySentry addDir;
  rootFile_->cd();

  LogDebug("Calibration") << "[DTResidualCalibration] Booking histos for layer: " << layerId;

  // Compose the chamber name
  std::string wheelStr = std::to_string(layerId.wheel());
  std::string stationStr = std::to_string(layerId.station());
  std::string sectorStr = std::to_string(layerId.sector());
  std::string superLayerStr = std::to_string(layerId.superlayer());
  std::string layerStr = std::to_string(layerId.layer());
  // Define the step
  int step = 3;

  std::string layerHistoName = "_STEP" + std::to_string(step) + "_W" + wheelStr + "_St" + stationStr + "_Sec" +
                               sectorStr + "_SL" + superLayerStr + "_Layer" + layerStr;

  LogDebug("Calibration") << "Accessing " << rootBaseDir_;
  TDirectory* baseDir = rootFile_->GetDirectory(rootBaseDir_.c_str());
  if (!baseDir)
    baseDir = rootFile_->mkdir(rootBaseDir_.c_str());
  LogDebug("Calibration") << "Accessing " << ("Wheel" + wheelStr);
  TDirectory* wheelDir = baseDir->GetDirectory(("Wheel" + wheelStr).c_str());
  if (!wheelDir)
    wheelDir = baseDir->mkdir(("Wheel" + wheelStr).c_str());
  LogDebug("Calibration") << "Accessing " << ("Station" + stationStr);
  TDirectory* stationDir = wheelDir->GetDirectory(("Station" + stationStr).c_str());
  if (!stationDir)
    stationDir = wheelDir->mkdir(("Station" + stationStr).c_str());
  LogDebug("Calibration") << "Accessing " << ("Sector" + sectorStr);
  TDirectory* sectorDir = stationDir->GetDirectory(("Sector" + sectorStr).c_str());
  if (!sectorDir)
    sectorDir = stationDir->mkdir(("Sector" + sectorStr).c_str());
  LogDebug("Calibration") << "Accessing " << ("SL" + superLayerStr);
  TDirectory* superLayerDir = sectorDir->GetDirectory(("SL" + superLayerStr).c_str());
  if (!superLayerDir)
    superLayerDir = sectorDir->mkdir(("SL" + superLayerStr).c_str());

  superLayerDir->cd();
  // Create histograms
  TH1F* histosTH1F = new TH1F(("hResDist" + layerHistoName).c_str(),
                              "Residuals on the distance from wire (rec_hit - segm_extr) (cm)",
                              200,
                              -histRange_,
                              histRange_);
  TH2F* histosTH2F = new TH2F(("hResDistVsDist" + layerHistoName).c_str(),
                              "Residuals on the dist. (cm) from wire (rec_hit - segm_extr) vs dist. (cm)",
                              100,
                              0,
                              2.5,
                              200,
                              -histRange_,
                              histRange_);
  histoMapPerLayerTH1F_[layerId] = histosTH1F;
  histoMapPerLayerTH2F_[layerId] = histosTH2F;
}

// Fill a set of histograms for a given SL
void DTResidualCalibration::fillHistos(DTSuperLayerId slId, float distance, float residualOnDistance) {
  histoMapTH1F_[slId]->Fill(residualOnDistance);
  histoMapTH2F_[slId]->Fill(distance, residualOnDistance);
}

// Fill a set of histograms for a given layer
void DTResidualCalibration::fillHistos(DTLayerId layerId, float distance, float residualOnDistance) {
  histoMapPerLayerTH1F_[layerId]->Fill(residualOnDistance);
  histoMapPerLayerTH2F_[layerId]->Fill(distance, residualOnDistance);
}

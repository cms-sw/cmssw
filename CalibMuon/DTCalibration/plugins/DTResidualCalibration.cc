
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/02/02 13:30:02 $
 *  $Revision: 1.4 $
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

DTResidualCalibration::DTResidualCalibration(const edm::ParameterSet& pset):
  select_(pset),
  segment4DLabel_(pset.getParameter<edm::InputTag>("segment4DLabel")),
  rootBaseDir_(pset.getUntrackedParameter<std::string>("rootBaseDir","DT/Residuals")),
  detailedAnalysis_(pset.getUntrackedParameter<bool>("detailedAnalysis",false)) {

  edm::LogVerbatim("Calibration") << "[DTResidualCalibration] Constructor called.";

  std::string rootFileName = pset.getUntrackedParameter<std::string>("rootFileName","residuals.root");
  rootFile_ = new TFile(rootFileName.c_str(), "RECREATE");
  rootFile_->cd();
}

DTResidualCalibration::~DTResidualCalibration() {
  edm::LogVerbatim("Calibration") << "[DTResidualCalibration] Destructor called.";
}

void DTResidualCalibration::beginJob() {
  TH1::SetDefaultSumw2(true);
}

void DTResidualCalibration::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  
  // get the geometry
  edm::ESHandle<DTGeometry> dtGeomH; 
  setup.get<MuonGeometryRecord>().get(dtGeomH);
  dtGeom_ = dtGeomH.product();

  // Loop over all the chambers
  if(histoMapTH1F_.size() == 0) { 	 
     std::vector<DTChamber*>::const_iterator ch_it = dtGeom_->chambers().begin(); 	 
     std::vector<DTChamber*>::const_iterator ch_end = dtGeom_->chambers().end(); 	 
     for (; ch_it != ch_end; ++ch_it) { 	 
        std::vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 	 
        std::vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end(); 	 
        // Loop over the SLs 	 
        for(; sl_it != sl_end; ++sl_it) { 
           DTSuperLayerId slId = (*sl_it)->id();
           bookHistos(slId);
           if(detailedAnalysis_) {
	      std::vector<const DTLayer*>::const_iterator layer_it = (*sl_it)->layers().begin(); 	 
	      std::vector<const DTLayer*>::const_iterator layer_end = (*sl_it)->layers().end();
	      for(; layer_it != layer_end; ++layer_it) { 
		 DTLayerId layerId = (*layer_it)->id();
		 bookHistos(layerId);
              }
           }
        }
     }
  }
}

void DTResidualCalibration::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  rootFile_->cd();

  // Get the 4D rechits from the event
  edm::Handle<DTRecSegment4DCollection> segment4Ds;
  event.getByLabel(segment4DLabel_, segment4Ds);
 
  // Loop over segments by chamber
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for(chamberIdIt = segment4Ds->id_begin(); chamberIdIt != segment4Ds->id_end(); ++chamberIdIt){

     const DTChamber* chamber = dtGeom_->chamber(*chamberIdIt);

     // Get the range for the corresponding ChamberId
     DTRecSegment4DCollection::range range = segment4Ds->get((*chamberIdIt));
     // Loop over the rechits of this DetUnit
     for(DTRecSegment4DCollection::const_iterator segment  = range.first;
                                                  segment != range.second; ++segment){

        LogTrace("Calibration") << "Segment local pos (in chamber RF): " << (*segment).localPosition()
                                << "\nSegment global pos: " << chamber->toGlobal((*segment).localPosition());

        if( !select_(*segment, event, setup) ) continue;

        // Get all 1D RecHits at step 3 within the 4D segment
        std::vector<DTRecHit1D> recHits1D_S3;
  
        if( (*segment).hasPhi() ){
           const DTChamberRecSegment2D* phiSeg = (*segment).phiSegment();
           const std::vector<DTRecHit1D>& phiRecHits = phiSeg->specificRecHits();
           std::copy(phiRecHits.begin(), phiRecHits.end(), back_inserter(recHits1D_S3));
        }

        if( (*segment).hasZed() ){
           const DTSLRecSegment2D* zSeg = (*segment).zSegment();
           const std::vector<DTRecHit1D>& zRecHits = zSeg->specificRecHits();
           std::copy(zRecHits.begin(), zRecHits.end(), back_inserter(recHits1D_S3));
        }

        // Loop over 1D RecHit inside 4D segment
        for(std::vector<DTRecHit1D>::const_iterator recHit1D = recHits1D_S3.begin();
                                                    recHit1D != recHits1D_S3.end(); ++recHit1D) {
           const DTWireId wireId = recHit1D->wireId();

           float segmDistance = segmentToWireDistance(*recHit1D,*segment);
           if(segmDistance > 2.1) LogTrace("Calibration") << "WARNING: segment-wire distance: " << segmDistance;
           else                   LogTrace("Calibration") << "segment-wire distance: " << segmDistance;

           float residualOnDistance = DTRecHitSegmentResidual().compute(dtGeom_,*recHit1D,*segment);
           LogTrace("Calibration") << "Wire Id " << wireId << " residual on distance: " << residualOnDistance;

           fillHistos(wireId.superlayerId(), segmDistance, residualOnDistance);
           if(detailedAnalysis_) fillHistos(wireId.layerId(), segmDistance, residualOnDistance);
        }
     }
  }

}

float DTResidualCalibration::segmentToWireDistance(const DTRecHit1D& recHit1D, const DTRecSegment4D& segment){

  // Get the layer and the wire position
  const DTWireId wireId = recHit1D.wireId();
  const DTLayer* layer = dtGeom_->layer(wireId);
  float wireX = layer->specificTopology().wirePosition(wireId.wire());
      
  // Extrapolate the segment to the z of the wire
  // Get wire position in chamber RF
  // (y and z must be those of the hit to be coherent in the transf. of RF in case of rotations of the layer alignment)
  LocalPoint wirePosInLay(wireX,recHit1D.localPosition().y(),recHit1D.localPosition().z());
  GlobalPoint wirePosGlob = layer->toGlobal(wirePosInLay);
  const DTChamber* chamber = dtGeom_->chamber(wireId.layerId().chamberId());
  LocalPoint wirePosInChamber = chamber->toLocal(wirePosGlob);
      
  // Segment position at Wire z in chamber local frame
  LocalPoint segPosAtZWire = segment.localPosition()	+ segment.localDirection()*wirePosInChamber.z()/cos(segment.localDirection().theta());
      
  // Compute the distance of the segment from the wire
  int sl = wireId.superlayer();
  float segmDistance = -1;
  if(sl == 1 || sl == 3) segmDistance = fabs(wirePosInChamber.x() - segPosAtZWire.x());
  else if(sl == 2)       segmDistance =  fabs(segPosAtZWire.y() - wirePosInChamber.y());
     
  return segmDistance;
}

void DTResidualCalibration::endJob(){
  
  edm::LogVerbatim("Calibration") << "[DTResidualCalibration] Writing histos to file.";
  rootFile_->cd();
  rootFile_->Write();
  rootFile_->Close();

  /*std::map<DTSuperLayerId, std::vector<TH1F*> >::const_iterator itSlHistos = histoMapTH1F_.begin();
  std::map<DTSuperLayerId, std::vector<TH1F*> >::const_iterator itSlHistos_end = histoMapTH1F_.end(); 
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

  edm::LogVerbatim("Calibration") << "[DTResidualCalibration] Booking histos for SL: " << slId;

  // Compose the chamber name
  std::stringstream wheelStr; wheelStr << slId.wheel();	
  std::stringstream stationStr; stationStr << slId.station();	
  std::stringstream sectorStr; sectorStr << slId.sector();	
  std::stringstream superLayerStr; superLayerStr << slId.superlayer();
  // Define the step
  int step = 3;
  std::stringstream stepStr; stepStr << step;

  std::string slHistoName =
    "_STEP" + stepStr.str() +
    "_W" + wheelStr.str() +
    "_St" + stationStr.str() +
    "_Sec" + sectorStr.str() +
    "_SL" + superLayerStr.str();
  
  edm::LogVerbatim("Calibration") << "Accessing " << rootBaseDir_;
  TDirectory* baseDir = rootFile_->GetDirectory(rootBaseDir_.c_str());
  if(!baseDir) baseDir = rootFile_->mkdir(rootBaseDir_.c_str());
  edm::LogVerbatim("Calibration") << "Accessing " << ("Wheel" + wheelStr.str());
  TDirectory* wheelDir = baseDir->GetDirectory(("Wheel" + wheelStr.str()).c_str());
  if(!wheelDir) wheelDir = baseDir->mkdir(("Wheel" + wheelStr.str()).c_str());
  edm::LogVerbatim("Calibration") << "Accessing " << ("Station" + stationStr.str());
  TDirectory* stationDir = wheelDir->GetDirectory(("Station" + stationStr.str()).c_str());
  if(!stationDir) stationDir = wheelDir->mkdir(("Station" + stationStr.str()).c_str());
  edm::LogVerbatim("Calibration") << "Accessing " << ("Sector" + sectorStr.str());
  TDirectory* sectorDir = stationDir->GetDirectory(("Sector" + sectorStr.str()).c_str());
  if(!sectorDir) sectorDir = stationDir->mkdir(("Sector" + sectorStr.str()).c_str()); 

  /*std::string dirName = rootBaseDir_ + "/Wheel" + wheelStr.str() +
                                       "/Station" + stationStr.str() +
                                       "/Sector" + sectorStr.str();

  TDirectory* dir = rootFile_->GetDirectory(dirName.c_str());
  if(!dir) dir = rootFile_->mkdir(dirName.c_str());
  dir->cd();*/
  sectorDir->cd();
  // Create the monitor elements
  std::vector<TH1F*> histosTH1F;
  histosTH1F.push_back(new TH1F(("hResDist"+slHistoName).c_str(),
				 "Residuals on the distance from wire (rec_hit - segm_extr) (cm)",
				 200, -0.4, 0.4));
  std::vector<TH2F*> histosTH2F;
  histosTH2F.push_back(new TH2F(("hResDistVsDist"+slHistoName).c_str(),
				 "Residuals on the dist. (cm) from wire (rec_hit - segm_extr) vs dist. (cm)",
                                 100, 0, 2.5, 200, -0.4, 0.4));
  histoMapTH1F_[slId] = histosTH1F;
  histoMapTH2F_[slId] = histosTH2F;
}

void DTResidualCalibration::bookHistos(DTLayerId layerId) {
  TH1AddDirectorySentry addDir;
  rootFile_->cd();

  edm::LogVerbatim("Calibration") << "[DTResidualCalibration] Booking histos for layer: " << layerId;

  // Compose the chamber name
  std::stringstream wheelStr; wheelStr << layerId.wheel();
  std::stringstream stationStr; stationStr << layerId.station();
  std::stringstream sectorStr; sectorStr << layerId.sector();
  std::stringstream superLayerStr; superLayerStr << layerId.superlayer();
  std::stringstream layerStr; layerStr << layerId.layer();
  // Define the step
  int step = 3;
  std::stringstream stepStr; stepStr << step;

  std::string layerHistoName =
    "_STEP" + stepStr.str() +
    "_W" + wheelStr.str() +
    "_St" + stationStr.str() +
    "_Sec" + sectorStr.str() +
    "_SL" + superLayerStr.str() + 
    "_Layer" + layerStr.str();
  
  edm::LogVerbatim("Calibration") << "Accessing " << rootBaseDir_;
  TDirectory* baseDir = rootFile_->GetDirectory(rootBaseDir_.c_str());
  if(!baseDir) baseDir = rootFile_->mkdir(rootBaseDir_.c_str());
  edm::LogVerbatim("Calibration") << "Accessing " << ("Wheel" + wheelStr.str());
  TDirectory* wheelDir = baseDir->GetDirectory(("Wheel" + wheelStr.str()).c_str());
  if(!wheelDir) wheelDir = baseDir->mkdir(("Wheel" + wheelStr.str()).c_str());
  edm::LogVerbatim("Calibration") << "Accessing " << ("Station" + stationStr.str());
  TDirectory* stationDir = wheelDir->GetDirectory(("Station" + stationStr.str()).c_str());
  if(!stationDir) stationDir = wheelDir->mkdir(("Station" + stationStr.str()).c_str());
  edm::LogVerbatim("Calibration") << "Accessing " << ("Sector" + sectorStr.str());
  TDirectory* sectorDir = stationDir->GetDirectory(("Sector" + sectorStr.str()).c_str());
  if(!sectorDir) sectorDir = stationDir->mkdir(("Sector" + sectorStr.str()).c_str()); 
  edm::LogVerbatim("Calibration") << "Accessing " << ("SL" + superLayerStr.str());
  TDirectory* superLayerDir = sectorDir->GetDirectory(("SL" + superLayerStr.str()).c_str());
  if(!superLayerDir) superLayerDir = sectorDir->mkdir(("SL" + superLayerStr.str()).c_str()); 

  superLayerDir->cd();
  // Create histograms
  std::vector<TH1F*> histosTH1F;
  histosTH1F.push_back(new TH1F(("hResDist"+layerHistoName).c_str(),
				 "Residuals on the distance from wire (rec_hit - segm_extr) (cm)",
				 200, -0.4, 0.4));
  std::vector<TH2F*> histosTH2F;
  histosTH2F.push_back(new TH2F(("hResDistVsDist"+layerHistoName).c_str(),
				 "Residuals on the dist. (cm) from wire (rec_hit - segm_extr) vs dist. (cm)",
                                 100, 0, 2.5, 200, -0.4, 0.4));
  histoMapPerLayerTH1F_[layerId] = histosTH1F;
  histoMapPerLayerTH2F_[layerId] = histosTH2F;
}

// Fill a set of histograms for a given SL 
void DTResidualCalibration::fillHistos(DTSuperLayerId slId,
				       float distance,
				       float residualOnDistance) {
  std::vector<TH1F*> const& histosTH1F = histoMapTH1F_[slId];
  std::vector<TH2F*> const& histosTH2F = histoMapTH2F_[slId];                          
  histosTH1F[0]->Fill(residualOnDistance);
  histosTH2F[0]->Fill(distance, residualOnDistance);
}

// Fill a set of histograms for a given layer 
void DTResidualCalibration::fillHistos(DTLayerId layerId,
				       float distance,
				       float residualOnDistance) {
  std::vector<TH1F*> const& histosTH1F = histoMapPerLayerTH1F_[layerId];
  std::vector<TH2F*> const& histosTH2F = histoMapPerLayerTH2F_[layerId];                          
  histosTH1F[0]->Fill(residualOnDistance);
  histosTH2F[0]->Fill(distance, residualOnDistance);
}



/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 */

#include "DQM/DTMonitorModule/interface/DTCalibValidation.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Geometry
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"


#include <iterator>

using namespace edm;
using namespace std;


DTCalibValidation::DTCalibValidation(const ParameterSet& pset) {

  parameters = pset;

  //FR the following was previously in the beginJob

  // the name of the rechits collection at step 1
  recHits1DToken_ = consumes<DTRecHitCollection>(
      edm::InputTag(parameters.getUntrackedParameter<string>("recHits1DLabel")));
  // the name of the 2D segments
  segment2DToken_ = consumes<DTRecSegment2DCollection>(
      edm::InputTag(parameters.getUntrackedParameter<string>("segment2DLabel")));
  // the name of the 4D segments
  segment4DToken_ = consumes<DTRecSegment4DCollection>(
      edm::InputTag(parameters.getUntrackedParameter<string>("segment4DLabel")));
  // the counter of segments not used to compute residuals
  wrongSegment = 0;
  // the counter of segments used to compute residuals
  rightSegment = 0;
  // the analysis type
  detailedAnalysis = parameters.getUntrackedParameter<bool>("detailedAnalysis",false);

  nevent=0;

}


DTCalibValidation::~DTCalibValidation(){

  //FR the following was previously in the endJob

 LogVerbatim("DTCalibValidation") << "Segments used to compute residuals: " << rightSegment;
 LogVerbatim("DTCalibValidation") << "Segments not used to compute residuals: " << wrongSegment;

}

 void DTCalibValidation::dqmBeginRun(const edm::Run& run, const edm::EventSetup& setup) {

 // get the geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);

 }

void DTCalibValidation::analyze(const edm::Event& event, const edm::EventSetup& setup) {

  ++nevent;
  LogTrace("DTCalibValidation") << "[DTCalibValidation] Analyze #Run: " << event.id().run()
                                << " #Event: " << nevent;

  // RecHit mapping at Step 1 -------------------------------
  map<DTWireId,vector<DTRecHit1DPair> > recHitsPerWire_1S;

  // RecHit mapping at Step 2 ------------------------------
  map<DTWireId,vector<DTRecHit1D> > recHitsPerWire_2S;

  if(detailedAnalysis){
     LogTrace("DTCalibValidation") << "  -- DTRecHit S1: begin analysis:";
     // Get the rechit collection from the event
     Handle<DTRecHitCollection> dtRecHits;
     event.getByToken(recHits1DToken_, dtRecHits);
     recHitsPerWire_1S = map1DRecHitsPerWire(dtRecHits.product());

     LogTrace("DTCalibValidation") << "  -- DTRecHit S2: begin analysis:";
     // Get the 2D rechits from the event
     Handle<DTRecSegment2DCollection> segment2Ds;
     event.getByToken(segment2DToken_, segment2Ds);
     recHitsPerWire_2S =  map1DRecHitsPerWire(segment2Ds.product());
  }

  // RecHit mapping at Step 3 ---------------------------------
  LogTrace("DTCalibValidation") << "  -- DTRecHit S3: begin analysis:";
  // Get the 4D rechits from the event
  Handle<DTRecSegment4DCollection> segment4Ds;
  event.getByToken(segment4DToken_, segment4Ds);
  map<DTWireId,vector<DTRecHit1D> > recHitsPerWire_3S =  map1DRecHitsPerWire(segment4Ds.product());


  // Loop over all 4D segments
  for(DTRecSegment4DCollection::const_iterator segment = segment4Ds->begin();
      segment != segment4Ds->end();
      ++segment) {

    if(detailedAnalysis){
       LogTrace("DTCalibValidation") << "Anlysis on recHit at step 1";
       compute(dtGeom.product(), (*segment), recHitsPerWire_1S, 1);

       LogTrace("DTCalibValidation") << "Anlysis on recHit at step 2";
       compute(dtGeom.product(), (*segment), recHitsPerWire_2S, 2);
    }

    LogTrace("DTCalibValidation") << "Anlysis on recHit at step 3";
    compute(dtGeom.product(), (*segment), recHitsPerWire_3S, 3);
  }

}


// Return a map between DTRecHit1DPair and wireId
map<DTWireId, vector<DTRecHit1DPair> >
DTCalibValidation::map1DRecHitsPerWire(const DTRecHitCollection* dt1DRecHitPairs) {
  map<DTWireId, vector<DTRecHit1DPair> > ret;

  for(DTRecHitCollection::const_iterator rechit = dt1DRecHitPairs->begin();
      rechit != dt1DRecHitPairs->end(); ++rechit) {
    ret[(*rechit).wireId()].push_back(*rechit);
  }

  return ret;
}


// Return a map between DTRecHit1D at S2 and wireId
map<DTWireId, vector<DTRecHit1D> >
DTCalibValidation::map1DRecHitsPerWire(const DTRecSegment2DCollection* segment2Ds) {
  map<DTWireId, vector<DTRecHit1D> > ret;

  // Loop over all 2D segments
  for(DTRecSegment2DCollection::const_iterator segment = segment2Ds->begin();
      segment != segment2Ds->end();
      ++segment) {
    vector<DTRecHit1D> component1DHits= (*segment).specificRecHits();
    // Loop over all component 1D hits
    for(vector<DTRecHit1D>::const_iterator hit = component1DHits.begin();
        hit != component1DHits.end(); ++hit) {
      ret[(*hit).wireId()].push_back(*hit);
    }
  }
  return ret;
}

// Return a map between DTRecHit1D at S3 and wireId
map<DTWireId, std::vector<DTRecHit1D> >
DTCalibValidation::map1DRecHitsPerWire(const DTRecSegment4DCollection* segment4Ds) {
  map<DTWireId, vector<DTRecHit1D> > ret;
  // Loop over all 4D segments
  for(DTRecSegment4DCollection::const_iterator segment = segment4Ds->begin();
      segment != segment4Ds->end();
      ++segment) {
    // Get component 2D segments
    vector<const TrackingRecHit*> segment2Ds = (*segment).recHits();
    // Loop over 2D segments:
    for(vector<const TrackingRecHit*>::const_iterator segment2D = segment2Ds.begin();
        segment2D != segment2Ds.end();
        ++segment2D) {
      // Get 1D component rechits
      vector<const TrackingRecHit*> hits = (*segment2D)->recHits();
      // Loop over them
      for(vector<const TrackingRecHit*>::const_iterator hit = hits.begin();
          hit != hits.end(); ++hit) {
        const DTRecHit1D* hit1D = dynamic_cast<const DTRecHit1D*>(*hit);
        ret[hit1D->wireId()].push_back(*hit1D);
      }
    }
  }

  return ret;
}



// Find the RecHit closest to the segment4D
template  <typename type>
const type*
DTCalibValidation::findBestRecHit(const DTLayer* layer,
                                DTWireId wireId,
                                const vector<type>& recHits,
                                const float segmDist) {
  float res = 99999;
  const type* theBestRecHit = 0;
  // Loop over RecHits within the cell
  for(typename vector<type>::const_iterator recHit = recHits.begin();
      recHit != recHits.end();
      ++recHit) {
    float distTmp = recHitDistFromWire(*recHit, layer);
    if(fabs(distTmp-segmDist) < res) {
      res = fabs(distTmp-segmDist);
      theBestRecHit = &(*recHit);
    }
  } // End of loop over RecHits within the cell

  return theBestRecHit;
}


// Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
float
DTCalibValidation::recHitDistFromWire(const DTRecHit1DPair& hitPair, const DTLayer* layer) {
  return fabs(hitPair.localPosition(DTEnums::Left).x() -
              hitPair.localPosition(DTEnums::Right).x())/2.;
}


// Compute the distance from wire (cm) of a hits in a DTRecHit1D
float
DTCalibValidation::recHitDistFromWire(const DTRecHit1D& recHit, const DTLayer* layer) {
  return fabs(recHit.localPosition().x() - layer->specificTopology().wirePosition(recHit.wireId().wire()));

}

// Compute the position (cm) of a hits in a DTRecHit1DPair
float
DTCalibValidation::recHitPosition(const DTRecHit1DPair& hitPair, const DTLayer* layer, const DTChamber* chamber, float segmentPos, int sl) {

  // Get the layer and the wire position
  GlobalPoint hitPosGlob_right = layer->toGlobal(hitPair.localPosition(DTEnums::Right));
  LocalPoint hitPosInChamber_right = chamber->toLocal(hitPosGlob_right);
  GlobalPoint hitPosGlob_left = layer->toGlobal(hitPair.localPosition(DTEnums::Left));
  LocalPoint hitPosInChamber_left = chamber->toLocal(hitPosGlob_left);

  float recHitPos=-1;
  if(sl != 2){
    if(fabs(hitPosInChamber_left.x()-segmentPos)<fabs(hitPosInChamber_right.x()-segmentPos))
      recHitPos = hitPosInChamber_left.x();
    else
      recHitPos = hitPosInChamber_right.x();
  }
  else{
    if(fabs(hitPosInChamber_left.y()-segmentPos)<fabs(hitPosInChamber_right.y()-segmentPos))
      recHitPos = hitPosInChamber_left.y();
    else
      recHitPos = hitPosInChamber_right.y();
  }

  return recHitPos;
}


// Compute the position (cm) of a hits in a  DTRecHit1D
float
DTCalibValidation::recHitPosition(const DTRecHit1D& recHit, const DTLayer* layer, const DTChamber* chamber, float segmentPos, int sl) {

  // Get the layer and the wire position
  GlobalPoint recHitPosGlob = layer->toGlobal(recHit.localPosition());
  LocalPoint recHitPosInChamber = chamber->toLocal(recHitPosGlob);

  float recHitPos = -1;
  if(sl != 2)
    recHitPos = recHitPosInChamber.x();
  else
    recHitPos = recHitPosInChamber.y();

  return recHitPos;

}

// Compute the residuals
template  <typename type>
void DTCalibValidation::compute(const DTGeometry *dtGeom,
                              const DTRecSegment4D& segment,
                              const std::map<DTWireId, std::vector<type> >& recHitsPerWire,
                              int step) {
  bool computeResidual = true;

  // Get all 1D RecHits at step 3 within the 4D segment
  vector<DTRecHit1D> recHits1D_S3;

  // Get 1D RecHits at Step 3 and select only events with
  // 8 hits in phi and 4 hits in theta (if any)
  const DTChamberRecSegment2D* phiSeg = segment.phiSegment();
  if(phiSeg){
    vector<DTRecHit1D> phiRecHits = phiSeg->specificRecHits();
    if(phiRecHits.size() != 8) {
      LogTrace("DTCalibValidation") << "[DTCalibValidation] Phi segments has: " << phiRecHits.size()
                                    << " hits, skipping"; // FIXME: info output
      computeResidual = false;
    }
    copy(phiRecHits.begin(), phiRecHits.end(), back_inserter(recHits1D_S3));
  }
  if(!phiSeg){
    LogTrace("DTCalibValidation") << " [DTCalibValidation] 4D segment has not the phi segment! ";
    computeResidual = false;
  }

  if(segment.dimension() == 4) {
    const DTSLRecSegment2D* zSeg = segment.zSegment();
    if(zSeg){
      vector<DTRecHit1D> zRecHits = zSeg->specificRecHits();
      if(zRecHits.size() != 4) {
        LogTrace("DTCalibValidation") << "[DTCalibValidation] Theta segments has: " << zRecHits.size()
	                              << " hits, skipping"; // FIXME: info output
	computeResidual = false;
      }
      copy(zRecHits.begin(), zRecHits.end(), back_inserter(recHits1D_S3));
    }
    if(!zSeg){
      LogTrace("DTCalibValidation") << " [DTCalibValidation] 4D segment has not the z segment! ";
      computeResidual = false;
    }
  }

  if(!computeResidual)
    ++wrongSegment;
  if(computeResidual){
    ++rightSegment;
    // Loop over 1D RecHit inside 4D segment
    for(vector<DTRecHit1D>::const_iterator recHit1D = recHits1D_S3.begin();
	recHit1D != recHits1D_S3.end();
	++recHit1D) {
      const DTWireId wireId = (*recHit1D).wireId();

      // Get the layer and the wire position
      const DTLayer* layer = dtGeom->layer(wireId);
      float wireX = layer->specificTopology().wirePosition(wireId.wire());

      // Extrapolate the segment to the z of the wire
      // Get wire position in chamber RF
      // (y and z must be those of the hit to be coherent in the transf. of RF in case of rotations of the layer alignment)
      LocalPoint wirePosInLay(wireX,(*recHit1D).localPosition().y(),(*recHit1D).localPosition().z());
      GlobalPoint wirePosGlob = layer->toGlobal(wirePosInLay);
      const DTChamber* chamber = dtGeom->chamber((*recHit1D).wireId().layerId().chamberId());
      LocalPoint wirePosInChamber = chamber->toLocal(wirePosGlob);

      // Segment position at Wire z in chamber local frame
      LocalPoint segPosAtZWire = segment.localPosition()
	+ segment.localDirection()*wirePosInChamber.z()/cos(segment.localDirection().theta());

      // Compute the distance of the segment from the wire
      int sl = wireId.superlayer();
      float SegmDistance = -1;
      if(sl == 1 || sl == 3) {
	// RPhi SL
	SegmDistance = fabs(wirePosInChamber.x() - segPosAtZWire.x());
	LogTrace("DTCalibValidation") << "SegmDistance: " << SegmDistance;
      } else if(sl == 2) {
	// RZ SL
	SegmDistance =  fabs(segPosAtZWire.y() - wirePosInChamber.y());
	LogTrace("DTCalibValidation") << "SegmDistance: " << SegmDistance;
      }
      if(SegmDistance > 2.1)
	LogTrace("DTCalibValidation") << "  Warning: dist segment-wire: " << SegmDistance;

      // Look for RecHits in the same cell
      if(recHitsPerWire.find(wireId) == recHitsPerWire.end()) {
        LogTrace("DTCalibValidation") << "   No RecHit found at Step: " << step << " in cell: " << wireId;
      } else {
	vector<type> recHits = recHitsPerWire.at(wireId);
	LogTrace("DTCalibValidation") << "   " << recHits.size() << " RecHits, Step " << step << " in channel: " << wireId;

	// Get the layer
	const DTLayer* layer = dtGeom->layer(wireId);
	// Find the best RecHits
	const type* theBestRecHit = findBestRecHit(layer, wireId, recHits, SegmDistance);
	// Compute the distance of the recHit from the wire
	float recHitWireDist =  recHitDistFromWire(*theBestRecHit, layer);
	LogTrace("DTCalibValidation") << "recHitWireDist: " << recHitWireDist;

	// Compute the residuals
	float residualOnDistance = recHitWireDist - SegmDistance;
	LogTrace("DTCalibValidation") << "WireId: " << wireId << "  ResidualOnDistance: " << residualOnDistance;
 	float residualOnPosition = -1;
	float recHitPos = -1;
	if(sl == 1 || sl == 3) {
	  recHitPos = recHitPosition(*theBestRecHit, layer, chamber,  segPosAtZWire.x(), sl);
	  residualOnPosition = recHitPos - segPosAtZWire.x();
	}
	else{
	  recHitPos = recHitPosition(*theBestRecHit, layer, chamber,  segPosAtZWire.y(), sl);
	  residualOnPosition = recHitPos - segPosAtZWire.y();
	}
	LogTrace("DTCalibValidation") << "WireId: " << wireId << "  ResidualOnPosition: " << residualOnPosition;

	// Fill the histos
	if(sl == 1 || sl == 3)
	  fillHistos(wireId.superlayerId(), SegmDistance, residualOnDistance, (wirePosInChamber.x() - segPosAtZWire.x()), residualOnPosition, step);
	else
	  fillHistos(wireId.superlayerId(), SegmDistance, residualOnDistance, (wirePosInChamber.y() - segPosAtZWire.y()), residualOnPosition, step);

      }
    }
  }

}


void DTCalibValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & iSetup) {

  //FR substitute the DQMStore instance by ibooker
   ibooker.setCurrentFolder("DT/DTCalibValidation");

  DTSuperLayerId slId;

  // Loop over all the chambers
  vector<const DTChamber*>::const_iterator ch_it = dtGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = dtGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin();
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();
    // Loop over the SLs
    for(; sl_it != sl_end; ++sl_it) {
      slId = (*sl_it)->id();

      int firstStep=1;
      if(!detailedAnalysis) firstStep=3;
      // Loop over the 3 steps
      for(int step = firstStep; step <= 3; ++step) {

         LogTrace("DTCalibValidation") << "   Booking histos for SL: " << slId;

         // Compose the chamber name
         stringstream wheel; wheel << slId.wheel();
         stringstream station; station << slId.station();
         stringstream sector; sector << slId.sector();
         stringstream superLayer; superLayer << slId.superlayer();
         // Define the step
         stringstream Step; Step << step;


         string slHistoName =
         "_STEP" + Step.str() +
         "_W" + wheel.str() +
         "_St" + station.str() +
         "_Sec" + sector.str() +
         "_SL" + superLayer.str();

         ibooker.setCurrentFolder("DT/DTCalibValidation/Wheel" + wheel.str() +
	          		   "/Station" + station.str() +
		        	   "/Sector" + sector.str());
         // Create the monitor elements
         vector<MonitorElement *> histos;
         // Note the order matters
          histos.push_back(ibooker.book1D("hResDist"+slHistoName,
	         			  "Residuals on the distance from wire (rec_hit - segm_extr) (cm)",
		        		  200, -0.4, 0.4));
          histos.push_back(ibooker.book2D("hResDistVsDist"+slHistoName,
	             			  "Residuals on the distance (cm) from wire (rec_hit - segm_extr) vs distance  (cm)",
		        		  100, 0, 2.5, 200, -0.4, 0.4));
          if(detailedAnalysis){
                          histos.push_back(ibooker.book1D("hResPos"+slHistoName,
	            	    "Residuals on the position from wire (rec_hit - segm_extr) (cm)",
	       		    200, -0.4, 0.4));
                         histos.push_back(ibooker.book2D("hResPosVsPos"+slHistoName,
		      	    "Residuals on the position (cm) from wire (rec_hit - segm_extr) vs distance  (cm)",
			    200, -2.5, 2.5, 200, -0.4, 0.4));
	  }

          histosPerSL[make_pair(slId, step)] = histos;
      }
    }
  }
}


// Fill a set of histograms for a given SL
void DTCalibValidation::fillHistos(DTSuperLayerId slId,
				     float distance,
				     float residualOnDistance,
				     float position,
				     float residualOnPosition,
				     int step) {
  // FIXME: optimization of the number of searches
  vector<MonitorElement *> histos =  histosPerSL[make_pair(slId,step)];
  histos[0]->Fill(residualOnDistance);
  histos[1]->Fill(distance, residualOnDistance);
  if(detailedAnalysis){
    histos[2]->Fill(residualOnPosition);
    histos[3]->Fill(position, residualOnPosition);
  }
}


// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:

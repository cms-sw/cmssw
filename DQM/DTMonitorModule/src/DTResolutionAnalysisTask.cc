
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/02/17 16:04:49 $
 *  $Revision: 1.23 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTResolutionAnalysisTask.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Geometry
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"


#include <iterator>

using namespace edm;
using namespace std;

DTResolutionAnalysisTask::DTResolutionAnalysisTask(const ParameterSet& pset) {

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionAnalysisTask] Constructor called!" << endl;

  // the name of the 4D rec hits collection
  theRecHits4DLabel = pset.getParameter<string>("recHits4DLabel");
  
  prescaleFactor = pset.getUntrackedParameter<int>("diagnosticPrescale", 1);
  resetCycle = pset.getUntrackedParameter<int>("ResetCycle", -1);
  // top folder for the histograms in DQMStore
  topHistoFolder = pset.getUntrackedParameter<string>("topHistoFolder","DT/02-Segments");

  thePhiHitsCut = pset.getUntrackedParameter<u_int32_t>("phiHitsCut",8);
  theZHitsCut = pset.getUntrackedParameter<u_int32_t>("zHitsCut",4);

}


DTResolutionAnalysisTask::~DTResolutionAnalysisTask(){

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionAnalysisTask] Destructor called!" << endl;

}

void DTResolutionAnalysisTask::beginRun(const Run& run, const EventSetup& setup) {

  // Get the DQM needed services
  theDbe = edm::Service<DQMStore>().operator->();
  //   theDbe->setCurrentFolder("DT/02-Segments");

  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Book the histograms
  vector<DTChamber*> chambers = dtGeom->chambers();
  for(vector<DTChamber*>::const_iterator chamber = chambers.begin();
      chamber != chambers.end(); ++chamber) {  // Loop over all chambers
    DTChamberId dtChId = (*chamber)->id();
    for(int sl = 1; sl <= 3; ++sl) { // Loop over SLs
      if(dtChId.station() == 4 && sl == 2) continue;
      const  DTSuperLayerId dtSLId(dtChId,sl);
      bookHistos(dtSLId);
    }
  }

}




void DTResolutionAnalysisTask::beginLuminosityBlock(const LuminosityBlock& lumiSeg,
						    const EventSetup& context) {

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionTask]: Begin of LS transition"<<endl;
  
  if(resetCycle != -1 && lumiSeg.id().luminosityBlock() % resetCycle == 0) {
    for(map<DTSuperLayerId, vector<MonitorElement*> > ::const_iterator histo = histosPerSL.begin();
	histo != histosPerSL.end();
	histo++) {
      int size = (*histo).second.size();
      for(int i=0; i<size; i++){
	(*histo).second[i]->Reset();
      }
    }
  }
}


void DTResolutionAnalysisTask::endJob(){

 edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionAnalysisTask] endjob called!"<<endl;

}
  


void DTResolutionAnalysisTask::analyze(const edm::Event& event, const edm::EventSetup& setup) {

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionAnalysisTask] Analyze #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;

  
  // Get the 4D segment collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel, all4DSegments);

  // check the validity of the collection 
  if(!all4DSegments.isValid()) return;

  // Loop over all chambers containing a segment
  DTRecSegment4DCollection::id_iterator chamberId;
  for (chamberId = all4DSegments->id_begin();
       chamberId != all4DSegments->id_end();
       ++chamberId) {
    // Get the range for the corresponding ChamerId
    DTRecSegment4DCollection::range  range = all4DSegments->get(*chamberId);
    //     int nsegm = distance(range.first, range.second);
    //edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "   Chamber: " << *chamberId << " has " << nsegm
    //<< " 4D segments" << endl;
    // Get the chamber
    const DTChamber* chamber = dtGeom->chamber(*chamberId);

    // Loop over the rechits of this ChamerId
    for (DTRecSegment4DCollection::const_iterator segment4D = range.first;
	 segment4D!=range.second;
	 ++segment4D) {
      //edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "   == RecSegment dimension: " << (*segment4D).dimension() << endl;
      
      // If Statio != 4 skip RecHits with dimension != 4
      // For the Station 4 consider 2D RecHits
      if((*chamberId).station() != 4 && (*segment4D).dimension() != 4) {
	edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionAnalysisTask]***Warning: RecSegment dimension is not 4 but "
	       << (*segment4D).dimension() << "!" << endl;
 	continue;
      } else if((*chamberId).station() == 4 && (*segment4D).dimension() != 2) {
	edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionAnalysisTask]***Warning: RecSegment dimension is not 2 but "
	       << (*segment4D).dimension() << "!" << endl;
 	continue;
      }


      // Get all 1D RecHits at step 3 within the 4D segment
      vector<DTRecHit1D> recHits1D_S3;
    

      // Get 1D RecHits at Step 3 and select only events with
      // 8 hits in phi and 4 hits in theta (if any)

      if((*segment4D).hasPhi()) { // has phi component
	const DTChamberRecSegment2D* phiSeg = (*segment4D).phiSegment();
	vector<DTRecHit1D> phiRecHits = phiSeg->specificRecHits();

	if(phiRecHits.size() < thePhiHitsCut) {
	  //edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionAnalysisTask] Phi segments has: " << phiRecHits.size()
	  //<< " hits" << endl; // FIXME: info output
	  continue;
	}
	copy(phiRecHits.begin(), phiRecHits.end(), back_inserter(recHits1D_S3));
      } else {
	//edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionAnalysisTask] 4D segment has not phi component!" << endl;
      }

      if((*segment4D).hasZed()) {
	const DTSLRecSegment2D* zSeg = (*segment4D).zSegment();
	vector<DTRecHit1D> zRecHits = zSeg->specificRecHits();
	if(zRecHits.size() < theZHitsCut) {
	  //edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "[DTResolutionAnalysisTask] Theta segments has: " << zRecHits.size()
	  //<< " hits, skipping" << endl; // FIXME: info output
 	  continue;
	}
	copy(zRecHits.begin(), zRecHits.end(), back_inserter(recHits1D_S3));
      }

      // Loop over 1D RecHit inside 4D segment
      for(vector<DTRecHit1D>::const_iterator recHit1D = recHits1D_S3.begin();
	  recHit1D != recHits1D_S3.end();
	  recHit1D++) {
	const DTWireId wireId = (*recHit1D).wireId();
	
	// Get the layer and the wire position
	const DTLayer* layer = chamber->superLayer(wireId.superlayerId())->layer(wireId.layerId());
	float wireX = layer->specificTopology().wirePosition(wireId.wire());

	// Distance of the 1D rechit from the wire
	float distRecHitToWire = fabs(wireX - (*recHit1D).localPosition().x());
	
	// Extrapolate the segment to the z of the wire
	
	// Get wire position in chamber RF
	LocalPoint wirePosInLay(wireX,(*recHit1D).localPosition().y(),(*recHit1D).localPosition().z());
	GlobalPoint wirePosGlob = layer->toGlobal(wirePosInLay);
	LocalPoint wirePosInChamber = chamber->toLocal(wirePosGlob);

	// Segment position at Wire z in chamber local frame
	LocalPoint segPosAtZWire = (*segment4D).localPosition()
	  + (*segment4D).localDirection()*wirePosInChamber.z()/cos((*segment4D).localDirection().theta());
	
	// Compute the distance of the segment from the wire
	int sl = wireId.superlayer();
  
	double distSegmToWire = -1;	
	if(sl == 1 || sl == 3) {
	  // RPhi SL
	  distSegmToWire = fabs(wirePosInChamber.x() - segPosAtZWire.x());
	} else if(sl == 2) {
	  // RZ SL
	  distSegmToWire = fabs(wirePosInChamber.y() - segPosAtZWire.y());
	}

	if(distSegmToWire > 2.1)
	  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "  Warning: dist segment-wire: " << distSegmToWire << endl;

	double residual = distRecHitToWire - distSegmToWire;
	// FIXME: Fill the histos
	fillHistos(wireId.superlayerId(), distSegmToWire, residual);
	
	//	edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "     Dist. segment extrapolation - wire (cm): " << distSegmToWire << endl;
	//edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "     Dist. RecHit - wire (cm): " << distRecHitToWire << endl;
	//edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "     Residual (cm): " << residual << endl;
	
			  
      }// End of loop over 1D RecHit inside 4D segment
    }// End of loop over the rechits of this ChamerId
  }
  // -----------------------------------------------------------------------------
}


// Book a set of histograms for a given SL
void DTResolutionAnalysisTask::bookHistos(DTSuperLayerId slId) {
  
  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask") << "   Booking histos for SL: " << slId << endl;

  // Compose the chamber name
  stringstream wheel; wheel << slId.wheel();	
  stringstream station; station << slId.station();	
  stringstream sector; sector << slId.sector();	
  stringstream superLayer; superLayer << slId.superlayer();	

  
  string slHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str() +
    "_SL" + superLayer.str();
  
  theDbe->setCurrentFolder(topHistoFolder + "/Wheel" + wheel.str() +
			   "/Sector" + sector.str() +
			   "/Station" + station.str());
  // Create the monitor elements
  vector<MonitorElement *> histos;
  // Note hte order matters
  histos.push_back(theDbe->book1D("hResDist"+slHistoName,
				  "Residuals on the distance from wire (rec_hit - segm_extr) (cm)",
				  200, -0.4, 0.4));
  //FIXME: 2D plot removed to reduce the # of ME
//   histos.push_back(theDbe->book2D("hResDistVsDist"+slHistoName,
// 				  "Residuals on the distance (cm) from wire (rec_hit - segm_extr) vs distance  (cm)",
// 				  100, 0, 2.5, 200, -0.4, 0.4));
  histosPerSL[slId] = histos;
}


// Fill a set of histograms for a given SL 
void DTResolutionAnalysisTask::fillHistos(DTSuperLayerId slId,
				      float distExtr,
				      float residual) {
  vector<MonitorElement *> histos =  histosPerSL[slId];                          
  histos[0]->Fill(residual);
  //FIXME: 2D plot removed to reduce the # of ME
  //   histos[1]->Fill(distExtr, residual); 

}




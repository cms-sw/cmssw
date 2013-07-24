

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/11/12 09:18:42 $
 *  $Revision: 1.17 $
 *  \author G. Mila - INFN Torino
 */


#include "DTEfficiencyTask.h"


// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Geometry
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRangeMapAccessor.h"

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include <iterator>

using namespace edm;
using namespace std;


DTEfficiencyTask::DTEfficiencyTask(const ParameterSet& pset) {

  debug = pset.getUntrackedParameter<bool>("debug",false);

  // Get the DQM needed services
  theDbe = edm::Service<DQMStore>().operator->();
  theDbe->setCurrentFolder("DT/DTEfficiencyTask");

  parameters = pset;
}


DTEfficiencyTask::~DTEfficiencyTask(){
}  


void DTEfficiencyTask::beginJob(){
  // the name of the 4D rec hits collection
  theRecHits4DLabel = parameters.getParameter<string>("recHits4DLabel");
  // the name of the rechits collection
  theRecHitLabel = parameters.getParameter<string>("recHitLabel");
}


void DTEfficiencyTask::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  
  if(lumiSeg.id().luminosityBlock()%parameters.getUntrackedParameter<int>("ResetCycle", 3) == 0) {
    for(map<DTLayerId, vector<MonitorElement*> > ::const_iterator histo = histosPerL.begin();
	histo != histosPerL.end();
	histo++) {
      int size = (*histo).second.size();
      for(int i=0; i<size; i++){
	(*histo).second[i]->Reset();
      }
    }
  }
  
}


void DTEfficiencyTask::endJob(){
  theDbe->rmdir("DT/DTEfficiencyTask");
}
  

void DTEfficiencyTask::analyze(const edm::Event& event, const edm::EventSetup& setup) {


  if(debug)
    cout << "[DTEfficiencyTask] Analyze #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;


  // Get the 4D segment collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel, all4DSegments);

  // Get the rechit collection from the event
  Handle<DTRecHitCollection> dtRecHits;
  event.getByLabel(theRecHitLabel, dtRecHits);

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Loop over all chambers containing a segment
  DTRecSegment4DCollection::id_iterator chamberId;
  for (chamberId = all4DSegments->id_begin();
       chamberId != all4DSegments->id_end();
       ++chamberId) {

    // Get the chamber
    const DTChamber* chamber = dtGeom->chamber(*chamberId); 

    // Get all 1D RecHits to be used for searches of hits not associated to segments and map them by wire
    const vector<const DTSuperLayer*> SLayers = chamber->superLayers();
    map<DTWireId, int> wireAnd1DRecHits;
    for(vector<const DTSuperLayer*>::const_iterator superlayer = SLayers.begin();
	superlayer != SLayers.end();
	superlayer++) {
	DTRecHitCollection::range  range = dtRecHits->get(DTRangeMapAccessor::layersBySuperLayer((*superlayer)->id()));
	// Loop over the rechits of this ChamberId
	for (DTRecHitCollection::const_iterator rechit = range.first;
	     rechit!=range.second;
	     ++rechit){
	  wireAnd1DRecHits[(*rechit).wireId()] = (*rechit).wireId().wire();
	}
      }


    // Get the 4D segment range for the corresponding ChamerId
    DTRecSegment4DCollection::range  range = all4DSegments->get(*chamberId);
    int nsegm = distance(range.first, range.second);
    if(debug)
      cout << "   Chamber: " << *chamberId << " has " << nsegm
	   << " 4D segments" << endl;

    
    // Loop over the rechits of this ChamerId
    for (DTRecSegment4DCollection::const_iterator segment4D = range.first;
	 segment4D!=range.second;
	 ++segment4D) {
      if(debug)
	cout << "   == RecSegment dimension: " << (*segment4D).dimension() << endl;

      // If Statio != 4 skip RecHits with dimension != 4
      // For the Station 4 consider 2D RecHits
      if((*chamberId).station() != 4 && (*segment4D).dimension() != 4) {
	if(debug)
	  cout << "[DTEfficiencyTask]***Warning: RecSegment dimension is not 4 but "
	       << (*segment4D).dimension() << ", skipping!" << endl;
	continue;
      } else if((*chamberId).station() == 4 && (*segment4D).dimension() != 2) {
	if(debug)
	  cout << "[DTEfficiencyTask]***Warning: RecSegment dimension is not 2 but "
	       << (*segment4D).dimension() << ", skipping!" << endl;
	continue;
      }

      vector<DTRecHit1D> recHits1D;   
      bool rPhi = false;
      bool rZ = false;

      // Get 1D RecHits and select only events with 7 or 8 hits in phi and 3 or 4 hits in theta (if any)
      const DTChamberRecSegment2D* phiSeg = (*segment4D).phiSegment();
      vector<DTRecHit1D> phiRecHits = phiSeg->specificRecHits();
      rPhi = true;
      if(phiRecHits.size() < 7 || phiRecHits.size() > 8 ) {
	if(debug)
	  cout << "[DTEfficiencyTask] Phi segments has: " << phiRecHits.size()
	       << " hits, skipping" << endl; // FIXME: info output
	continue;
      }
      copy(phiRecHits.begin(), phiRecHits.end(), back_inserter(recHits1D));
      const DTSLRecSegment2D* zSeg = 0;
      if((*segment4D).dimension() == 4) {
	rZ = true;
	zSeg = (*segment4D).zSegment();
	vector<DTRecHit1D> zRecHits = zSeg->specificRecHits();
	if(zRecHits.size() < 3 || zRecHits.size() > 4 ) {
	  if(debug)
	    cout << "[DTEfficiencyTask] Theta segments has: " << zRecHits.size()
		 << " hits, skipping" << endl; // FIXME: info output
	  continue;
	}
	copy(zRecHits.begin(), zRecHits.end(), back_inserter(recHits1D));
      }

      // Skip the segment if it has more than 1 hit on the same layer
      vector<DTWireId> wireMap; 
      for(vector<DTRecHit1D>::const_iterator recHit1D = recHits1D.begin();
	  recHit1D != recHits1D.end();
	  recHit1D++) {
	wireMap.push_back((*recHit1D).wireId());
      }

      bool hitsOnSameLayer = false;
      for(vector<DTWireId>::const_iterator channelId = wireMap.begin();
	  channelId != wireMap.end(); channelId++) {
	vector<DTWireId>::const_iterator next = channelId;
	next++;
	for(vector<DTWireId>::const_iterator ite = next; ite != wireMap.end(); ite++) {
	  if((*channelId).layerId() == (*ite).layerId()) {
	    hitsOnSameLayer = true;
	  }
	}
      }
      if(hitsOnSameLayer) {
	if(debug)
	  cout << "[DTEfficiencyTask] This RecHit has 2 hits on the same layer, skipping!" << endl;
	continue;
      }


      // Select 2D segments with angle smaller than 45 deg
      LocalVector phiDirectionInChamber = (*phiSeg).localDirection();
      if(rPhi && fabs(phiDirectionInChamber.x()/phiDirectionInChamber.z()) > 1) {
	if(debug) {
	  cout << "         RPhi segment has angle > 45 deg, skipping! " << endl;
	  cout << "              Theta = " << phiDirectionInChamber.theta() << endl;
	}
	continue;
      }
      if(rZ) {
	 LocalVector zDirectionInChamber = (*zSeg).localDirection();
	 if(fabs(zDirectionInChamber.y()/zDirectionInChamber.z()) > 1) {
	   if(debug){
	     cout << "         RZ segment has angle > 45 deg, skipping! "  << endl;
	     cout << "              Theta = " << zDirectionInChamber.theta() << endl;
	   }
	   continue;
	 }
      }


      // Skip if the 4D segment has only 10 hits
      if(recHits1D.size() == 10) {
	if(debug)
	  cout << "[DTEfficiencyTask] 4D Segment with only 10 hits, skipping!" << endl;
	continue;
      }


      // Analyse the case of 11 recHits for MB1,MB2,MB3 and of 7 recHits for MB4
      if((rPhi && recHits1D.size() == 7) || (rZ && recHits1D.size() == 11)) {

	if(debug) {
	  if(rPhi && recHits1D.size() == 7)
	    cout << "[DTEfficiencyTask] MB4 Segment with only 7 hits!" << endl;
	  if(rZ && recHits1D.size() == 11)
	     cout << "[DTEfficiencyTask] 4D Segment with only 11 hits!" << endl;
	}

	// Find the layer without RecHits ----------------------------------------
	const vector<const DTSuperLayer*> SupLayers = chamber->superLayers();
        map<DTLayerId, bool> layerMap; 
	map<DTWireId, float> wireAndPosInChamberAtLayerZ;
	// Loop over layers and wires to fill a map
	for(vector<const DTSuperLayer*>::const_iterator superlayer = SupLayers.begin();
	    superlayer != SupLayers.end();
	    superlayer++) {
	  const vector<const DTLayer*> Layers = (*superlayer)->layers();
	  for(vector<const DTLayer*>::const_iterator layer = Layers.begin();
	      layer != Layers.end();
	      layer++) {
	    layerMap.insert(make_pair((*layer)->id(), false));
	    const int firstWire = dtGeom->layer((*layer)->id())->specificTopology().firstChannel();
	    const int lastWire = dtGeom->layer((*layer)->id())->specificTopology().lastChannel();
	    for(int i=firstWire; i - lastWire <= 0; i++) {
	      DTWireId wireId((*layer)->id(), i);	     
	      float wireX = (*layer)->specificTopology().wirePosition(wireId.wire());
	      LocalPoint wirePosInLay(wireX,0,0);
	      GlobalPoint wirePosGlob = (*layer)->toGlobal(wirePosInLay);
	      LocalPoint wirePosInChamber = chamber->toLocal(wirePosGlob); 
	      if((*superlayer)->id().superlayer() == 1 || (*superlayer)->id().superlayer() == 3) {
		wireAndPosInChamberAtLayerZ.insert(make_pair(wireId, wirePosInChamber.x()));
	      } else {
		wireAndPosInChamberAtLayerZ.insert(make_pair(wireId, wirePosInChamber.y()));
	      }
	    }
	  }
	}

	// Loop over segment 1D RecHit
	map<DTLayerId, int> NumWireMap; 
	for(vector<DTRecHit1D>::const_iterator recHit = recHits1D.begin();
	    recHit != recHits1D.end();
	    recHit++) {
	  layerMap[(*recHit).wireId().layerId()]= true;
	  NumWireMap[(*recHit).wireId().layerId()]= (*recHit).wireId().wire();
	}

	DTLayerId missLayerId;
	//Loop over the map and find the layer without hits
	for(map<DTLayerId, bool>::const_iterator iter = layerMap.begin();
	    iter != layerMap.end(); iter++) {
	  if(!(*iter).second) missLayerId = (*iter).first;
	}
	if(debug)
	  cout << "[DTEfficiencyTask] Layer without recHits is: " << missLayerId << endl;
	// -------------------------------------------------------



	const DTLayer* missLayer = chamber->layer(missLayerId);
	
	LocalPoint missLayerPosInChamber = chamber->toLocal(missLayer->toGlobal(LocalPoint(0,0,0)));
	
	// Segment position at Wire z in chamber local frame
	LocalPoint segPosAtZLayer = (*segment4D).localPosition()
	  + (*segment4D).localDirection()*missLayerPosInChamber.z()/cos((*segment4D).localDirection().theta());
	
	DTWireId missWireId;

	// Find the id of the cell without hit ---------------------------------------------------
	for(map<DTWireId, float>::const_iterator wireAndPos = wireAndPosInChamberAtLayerZ.begin();
	    wireAndPos != wireAndPosInChamberAtLayerZ.end();
	    wireAndPos++) {
	  DTWireId wireId = (*wireAndPos).first;
	  if(wireId.layerId() == missLayerId) {
	    if(missLayerId.superlayerId().superlayer() == 1 || missLayerId.superlayerId().superlayer() == 3 ) {
	      if(fabs(segPosAtZLayer.x() - (*wireAndPos).second) < 2.1)
		missWireId = wireId;  
	    } else {
	      if(fabs(segPosAtZLayer.y() - (*wireAndPos).second) < 2.1)
		missWireId = wireId;
	    }
	  }
	}
	if(debug)
	  cout << "[DTEfficiencyTask] Cell without hit is: " << missWireId << endl;
	// ----------------------------------------------------------


	bool foundUnAssRechit = false;

	// Look for unassociated hits in this cell
	if(wireAnd1DRecHits.find(missWireId) != wireAnd1DRecHits.end()) {
	  if(debug)
	    cout << "[DTEfficiencyTask] Unassociated Hit found!" << endl;
	  foundUnAssRechit = true;
	}


	for(map<DTLayerId, bool>::const_iterator iter = layerMap.begin();
	    iter != layerMap.end(); iter++) {
	  if((*iter).second) 
	    fillHistos((*iter).first, dtGeom->layer((*iter).first)->specificTopology().firstChannel(), dtGeom->layer((*iter).first)->specificTopology().lastChannel(), NumWireMap[(*iter).first]);
	  else
	    fillHistos((*iter).first, dtGeom->layer((*iter).first)->specificTopology().firstChannel(), dtGeom->layer((*iter).first)->specificTopology().lastChannel(), missWireId.wire(), foundUnAssRechit);
	}

      } // End of the loop for segment with 7 or 11 recHits
      
      if((rPhi && recHits1D.size() == 8) || (rZ && recHits1D.size() == 12)) {
	map<DTLayerId, int> NumWireMap; 
	DTLayerId LayerID;
	for(vector<DTRecHit1D>::const_iterator recHit = recHits1D.begin();
	    recHit != recHits1D.end();
	    recHit++) {
	  LayerID = (*recHit).wireId().layerId();
	  NumWireMap[LayerID]= (*recHit).wireId().wire();
	}
	for(map<DTLayerId, int>::const_iterator iter = NumWireMap.begin();
	    iter != NumWireMap.end(); iter++) {
	  fillHistos((*iter).first, dtGeom->layer((*iter).first)->specificTopology().firstChannel(), dtGeom->layer((*iter).first)->specificTopology().lastChannel(), NumWireMap[(*iter).first]);
	}
      }

    } // End of loop over the 4D segments inside a sigle chamber
  } // End of loop over all tha chamber with at least a 4D segment in the event
}


// Book a set of histograms for a given Layer
void DTEfficiencyTask::bookHistos(DTLayerId lId, int firstWire, int lastWire) {
  if(debug)
    cout << "   Booking histos for L: " << lId << endl;

  // Compose the chamber name
  stringstream wheel; wheel << lId.superlayerId().chamberId().wheel();	
  stringstream station; station << lId.superlayerId().chamberId().station();	
  stringstream sector; sector << lId.superlayerId().chamberId().sector();	
  stringstream superLayer; superLayer << lId.superlayerId().superlayer();	
  stringstream layer; layer << lId.layer();

  string lHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str() +
    "_SL" + superLayer.str()+
    "_L" + layer.str();
  
  theDbe->setCurrentFolder("DT/DTEfficiencyTask/Wheel" + wheel.str() +
			   "/Station" + station.str() +
			   "/Sector" + sector.str() +
			   "/SuperLayer" +superLayer.str());
  // Create the monitor elements
  vector<MonitorElement *> histos;
  // histo for hits associated to the 4D reconstructed segment
  histos.push_back(theDbe->book1D("hEffOccupancy"+lHistoName, "4D segments recHits occupancy",lastWire-firstWire+1, firstWire-0.5, lastWire+0.5));
  // histo for hits not associated to the segment
  histos.push_back(theDbe->book1D("hEffUnassOccupancy"+lHistoName, "4D segments recHits and Hits not associated occupancy",lastWire-firstWire+1, firstWire-0.5, lastWire+0.5));
  // histo for cells associated to the 4D reconstructed segment
  histos.push_back(theDbe->book1D("hRecSegmOccupancy"+lHistoName, "4D segments cells occupancy",lastWire-firstWire+1, firstWire-0.5, lastWire+0.5));
  
  histosPerL[lId] = histos;
}


// Fill a set of histograms for a given Layer 
void DTEfficiencyTask::fillHistos(DTLayerId lId,
				  int firstWire, int lastWire,
				  int numWire) {
  if(histosPerL.find(lId) == histosPerL.end()){
      bookHistos(lId, firstWire, lastWire);
  }
  vector<MonitorElement *> histos =  histosPerL[lId]; 
  histos[0]->Fill(numWire);
  histos[1]->Fill(numWire);
  histos[2]->Fill(numWire);
}

// Fill a set of histograms for a given Layer
void DTEfficiencyTask::fillHistos(DTLayerId lId,
				  int firstWire, int lastWire,
				  int missingWire,
				  bool unassHit) {
 if(histosPerL.find(lId) == histosPerL.end()){
      bookHistos(lId, firstWire, lastWire);
  }
 vector<MonitorElement *> histos =  histosPerL[lId];
 if(unassHit) 
   histos[1]->Fill(missingWire);
 histos[2]->Fill(missingWire);
}

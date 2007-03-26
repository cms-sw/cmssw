

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/03/26 17:30:00 $
 *  $Revision: 1.0 $
 *  \author G. Mila - INFN Torino
 */


#include "DTEfficiencyTask.h"


// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

//Geometry
#include "Geometry/Vector/interface/Pi.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include <iterator>

using namespace edm;
using namespace std;


DTEfficiencyTask::DTEfficiencyTask(const ParameterSet& pset) {

  debug = pset.getUntrackedParameter<bool>("debug","false");
  if(debug)
    cout << "[DTEfficiencyTask] Constructor called!" << endl;

  // Get the DQM needed services
  theDbe = edm::Service<DaqMonitorBEInterface>().operator->();
  theDbe->setVerbose(1);
  edm::Service<MonitorDaemon>().operator->();
  theDbe->setCurrentFolder("DT/DTEfficiencyTask");

  // set the name of the outputfile
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName","DTEfficiencyTask.root");
  writeHisto = pset.getUntrackedParameter<bool>("writeHisto", true);

  parameters = pset;
}


DTEfficiencyTask::~DTEfficiencyTask(){
  if(debug)
    cout << "[DTEfficiencyTask] Destructor called!" << endl;
}  


void DTEfficiencyTask::beginJob(const edm::EventSetup& context){
  // the name of the 4D rec hits collection
  theRecHits4DLabel = parameters.getParameter<string>("recHits4DLabel");
}

void DTEfficiencyTask::endJob(){
 if(debug)
    cout<<"[DTEfficiencyTask] endjob called!"<<endl;
  // Write the histos
  if ( writeHisto ) 
    theDbe->save(theRootFileName);
  theDbe->rmdir("DT/DTEfficiencyTask");
}
  

void DTEfficiencyTask::analyze(const edm::Event& event, const edm::EventSetup& setup) {

  if(debug)
    cout << "[DTEfficiencyTask] Analyze #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;


  // Get the 4D segment collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel, all4DSegments);

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Loop over all chambers containing a segment
  DTRecSegment4DCollection::id_iterator chamberId;
  for (chamberId = all4DSegments->id_begin();
       chamberId != all4DSegments->id_end();
       ++chamberId) {

    // Get the range for the corresponding ChamerId
    DTRecSegment4DCollection::range  range = all4DSegments->get(*chamberId);
    int nsegm = distance(range.first, range.second);
    if(debug)
      cout << "   Chamber: " << *chamberId << " has " << nsegm
	   << " 4D segments" << endl;

    // Get the chamber
    const DTChamber* chamber = dtGeom->chamber(*chamberId);
  
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

      // Get 1D RecHits and select only events with
      // 7 or 8 hits in phi and 3 or 4 hits in theta (if any)
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
      if((*segment4D).dimension() == 4) {
	rZ = true;
	const DTSLRecSegment2D* zSeg = (*segment4D).zSegment();
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

      // Skip if the segment has only 6 hits
      if(recHits1D.size() == 6) {
	if(debug)
	  cout << "[DTEfficiencyTask] Segment with only 6 hits, skipping!" << endl;
	continue;
      }

      if(recHits1D.size() == 7) {
	if(debug)
	  cout << "[DTEfficiencyTask] Segment with only 7 hits!" << endl;

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
	    for(int i=firstWire; i <= lastWire; i++) {
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
	for(vector<DTRecHit1D>::const_iterator recHit = recHits1D.begin();
	  recHit != recHits1D.end();
	  recHit++) {
	  
	  layerMap[DTLayerId((*recHit).wireId().layerId().superlayerId().chamberId().wheel(),
			  (*recHit).wireId().layerId().superlayerId().chamberId().station(),
			  (*recHit).wireId().layerId().superlayerId().chamberId().sector(),
			  (*recHit).wireId().layerId().superlayerId().superlayer(),
			  (*recHit).wireId().layerId().layer())] = true;
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



	const DTLayer* missLayer = chamber->superLayer(missLayerId.superlayerId())->layer(missLayerId);
	
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

	for(map<DTLayerId, bool>::const_iterator iter = layerMap.begin();
	    iter != layerMap.end(); iter++) {
	  if((*iter).second) 
	    fillHistos((*iter).first, dtGeom->layer((*iter).first)->specificTopology().firstChannel(), dtGeom->layer((*iter).first)->specificTopology().lastChannel());
	  else
	    fillHistos((*iter).first, dtGeom->layer((*iter).first)->specificTopology().firstChannel(), dtGeom->layer((*iter).first)->specificTopology().lastChannel(), missWireId.wire());
	}
      }

      if(recHits1D.size() == 8) {
	const vector<const DTSuperLayer*> SupLayers = chamber->superLayers();
	for(vector<const DTSuperLayer*>::const_iterator superlayer = SupLayers.begin();
	    superlayer != SupLayers.end();
	    superlayer++) {
	  const vector<const DTLayer*> Layers = (*superlayer)->layers();
	  for(vector<const DTLayer*>::const_iterator layer = Layers.begin();
	      layer != Layers.end();
	      layer++) {
	    fillHistos((*layer)->id(), dtGeom->layer((*layer)->id())->specificTopology().firstChannel(), dtGeom->layer((*layer)->id())->specificTopology().lastChannel());
	  }
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
  // Note hte order matters
  histos.push_back(theDbe->book1D("hEfficiency"+lHistoName, "Efficiency per cell",lastWire-firstWire+1, firstWire-0.5, lastWire+0.5));

  histosPerL[lId] = histos;
}


// Fill a set of histograms for a given Layer 
void DTEfficiencyTask::fillHistos(DTLayerId lId,
				  int firstWire, int lastWire) {
  if(histosPerL.find(lId) == histosPerL.end()){
      bookHistos(lId, firstWire, lastWire);
  }
  vector<MonitorElement *> histos =  histosPerL[lId]; 
  for(int i=firstWire; i <= lastWire; i++) {
      histos[0]->Fill(i);
  }
}

// Fill a set of histograms for a given Layer
void DTEfficiencyTask::fillHistos(DTLayerId lId,
				  int firstWire, int lastWire,
				  int missingWire) {
 if(histosPerL.find(lId) == histosPerL.end()){
      bookHistos(lId, firstWire, lastWire);
  }
 vector<MonitorElement *> histos =  histosPerL[lId]; 
 for(int i=firstWire; i <= lastWire; i++) {
   if( i != missingWire )
     histos[0]->Fill(i);
 }
}

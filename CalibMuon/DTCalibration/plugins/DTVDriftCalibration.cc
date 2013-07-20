
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/05/23 15:28:45 $
 *  $Revision: 1.12 $
 *  \author M. Giunta
 */

#include "CalibMuon/DTCalibration/plugins/DTVDriftCalibration.h"
#include "CalibMuon/DTCalibration/interface/DTMeanTimerFitter.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

/* C++ Headers */
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TROOT.h" 

// Declare histograms.
TH1F * hChi2;
extern void bookHistos();

using namespace std;
using namespace edm;
using namespace dttmaxenums;


DTVDriftCalibration::DTVDriftCalibration(const ParameterSet& pset): select_(pset) {

  // The name of the 4D rec hits collection
  theRecHits4DLabel = pset.getParameter<InputTag>("recHits4DLabel");

  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  debug = pset.getUntrackedParameter<bool>("debug", false);

  theFitter = new DTMeanTimerFitter(theFile);
  if(debug)
    theFitter->setVerbosity(1);

  hChi2 = new TH1F("hChi2","Chi squared tracks",100,0,100);
  h2DSegmRPhi = new h2DSegm("SLRPhi");
  h2DSegmRZ = new h2DSegm("SLRZ");
  h4DSegmAllCh = new h4DSegm("AllCh");
  bookHistos();

  findVDriftAndT0 = pset.getUntrackedParameter<bool>("fitAndWrite", false);

  // Chamber/s to calibrate
  theCalibChamber =  pset.getUntrackedParameter<string>("calibChamber", "All");

  // the txt file which will contain the calibrated constants
  theVDriftOutputFile = pset.getUntrackedParameter<string>("vDriftFileName");

  // Get the synchronizer
  theSync = DTTTrigSyncFactory::get()->create(pset.getParameter<string>("tTrigMode"),
                                              pset.getParameter<ParameterSet>("tTrigModeConfig"));

  // get parameter set for DTCalibrationMap constructor
  theCalibFilePar =  pset.getUntrackedParameter<ParameterSet>("calibFileConfig");

  // the granularity to be used for tMax
  string tMaxGranularity = pset.getUntrackedParameter<string>("tMaxGranularity","bySL");

  // Enforce it to be by SL since rest is not implemented
  if(tMaxGranularity != "bySL"){
     LogError("Calibration") << "[DTVDriftCalibration] tMaxGranularity will be fixed to bySL.";
     tMaxGranularity = "bySL";
  }
  // Initialize correctly the enum which specify the granularity for the calibration
  if(tMaxGranularity == "bySL") {
    theGranularity = bySL;
  } else if(tMaxGranularity == "byChamber"){
    theGranularity = byChamber;
  } else if(tMaxGranularity== "byPartition") {
    theGranularity = byPartition;
  } else throw cms::Exception("Configuration") << "[DTVDriftCalibration] Check parameter tMaxGranularity: "
	                                       << tMaxGranularity << " option not available";
  

  LogVerbatim("Calibration") << "[DTVDriftCalibration]Constructor called!";
}

DTVDriftCalibration::~DTVDriftCalibration(){
  theFile->Close();
  delete theFitter;
  LogVerbatim("Calibration") << "[DTVDriftCalibration]Destructor called!";
}

void DTVDriftCalibration::analyze(const Event & event, const EventSetup& eventSetup) {
  LogTrace("Calibration") << "--- [DTVDriftCalibration] Event analysed #Run: " << event.id().run()
                          << " #Event: " << event.id().event();
  theFile->cd();
  DTChamberId chosenChamberId;

  if(theCalibChamber != "All") {
    stringstream linestr;
    int selWheel, selStation, selSector;
    linestr << theCalibChamber;
    linestr >> selWheel >> selStation >> selSector;
    chosenChamberId = DTChamberId(selWheel, selStation, selSector);
    LogTrace("Calibration") << "chosen chamber " << chosenChamberId;
  }

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the rechit collection from the event
  Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel, all4DSegments); 

  // Get the map of noisy channels
  /*ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    eventSetup.get<DTStatusFlagRcd>().get(statusMap);
  }*/

  // Set the event setup in the Synchronizer 
  theSync->setES(eventSetup);

  // Loop over segments by chamber
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for (chamberIdIt = all4DSegments->id_begin();
       chamberIdIt != all4DSegments->id_end();
       ++chamberIdIt){

    // Get the chamber from the setup
    const DTChamber* chamber = dtGeom->chamber(*chamberIdIt);
    LogTrace("Calibration") << "Chamber Id: " << *chamberIdIt;

    // Calibrate just the chosen chamber/s    
    if((theCalibChamber != "All") && ((*chamberIdIt) != chosenChamberId)) 
      continue;

    // Get the range for the corresponding ChamberId
    DTRecSegment4DCollection::range  range = all4DSegments->get((*chamberIdIt));

    // Loop over the rechits of this DetUnit
    for (DTRecSegment4DCollection::const_iterator segment = range.first;
         segment!=range.second; ++segment){

      if( !(*segment).hasZed() && !(*segment).hasPhi() ){
         LogError("Calibration") << "4D segment without Z and Phi segments";
         continue;   
      } 

      LogTrace("Calibration") << "Segment local pos (in chamber RF): " << (*segment).localPosition()
                              << "\nSegment global pos: " << chamber->toGlobal((*segment).localPosition());

      if( !select_(*segment, event, eventSetup) ) continue;

      LocalPoint phiSeg2DPosInCham;  
      LocalVector phiSeg2DDirInCham;
      map<DTSuperLayerId,vector<DTRecHit1D> > hitsBySLMap; 
      if((*segment).hasPhi()){
        const DTChamberRecSegment2D* phiSeg = (*segment).phiSegment();  // phiSeg lives in the chamber RF
        phiSeg2DPosInCham = phiSeg->localPosition();  
        phiSeg2DDirInCham = phiSeg->localDirection();
        vector<DTRecHit1D> phiHits = phiSeg->specificRecHits();
        for(vector<DTRecHit1D>::const_iterator hit = phiHits.begin();
            hit != phiHits.end(); ++hit) {
          DTWireId wireId = (*hit).wireId();
          DTSuperLayerId slId =  wireId.superlayerId();
          hitsBySLMap[slId].push_back(*hit); 
        }
      }
      // Get the Theta 2D segment and plot the angle in the chamber RF
      LocalVector zSeg2DDirInCham;
      LocalPoint zSeg2DPosInCham;
      if((*segment).hasZed()) {
        const DTSLRecSegment2D* zSeg = (*segment).zSegment();  // zSeg lives in the SL RF
        const DTSuperLayer* sl = chamber->superLayer(zSeg->superLayerId());
        zSeg2DPosInCham = chamber->toLocal(sl->toGlobal((*zSeg).localPosition())); 
        zSeg2DDirInCham = chamber->toLocal(sl->toGlobal((*zSeg).localDirection()));
        hitsBySLMap[zSeg->superLayerId()] = zSeg->specificRecHits();
      } 

      LocalPoint segment4DLocalPos = (*segment).localPosition();
      LocalVector segment4DLocalDir = (*segment).localDirection();
      double chiSquare = ((*segment).chi2()/(*segment).degreesOfFreedom());

      hChi2->Fill(chiSquare);
      if((*segment).hasPhi())
        h2DSegmRPhi->Fill(phiSeg2DPosInCham.x(), phiSeg2DDirInCham.x()/phiSeg2DDirInCham.z());
      if((*segment).hasZed())
        h2DSegmRZ->Fill(zSeg2DPosInCham.y(), zSeg2DDirInCham.y()/zSeg2DDirInCham.z());

      if((*segment).hasZed() && (*segment).hasPhi()) 
        h4DSegmAllCh->Fill(segment4DLocalPos.x(), 
                           segment4DLocalPos.y(),
                           atan(segment4DLocalDir.x()/segment4DLocalDir.z())*180./Geom::pi(),
                           atan(segment4DLocalDir.y()/segment4DLocalDir.z())*180./Geom::pi(),
                           180 - segment4DLocalDir.theta()*180./Geom::pi());
      else if((*segment).hasPhi())
        h4DSegmAllCh->Fill(segment4DLocalPos.x(), 
                           atan(segment4DLocalDir.x()/segment4DLocalDir.z())*180./Geom::pi());
      else if((*segment).hasZed())
        LogWarning("Calibration") << "4d segment with only Z";

      //loop over the segments 
      for(map<DTSuperLayerId,vector<DTRecHit1D> >::const_iterator slIdAndHits = hitsBySLMap.begin(); slIdAndHits != hitsBySLMap.end();  ++slIdAndHits) {
        if (slIdAndHits->second.size() < 3) continue;
        DTSuperLayerId slId = slIdAndHits->first;

        // Create the DTTMax, that computes the 4 TMax
        DTTMax slSeg(slIdAndHits->second, *(chamber->superLayer(slIdAndHits->first)),chamber->toGlobal((*segment).localDirection()), chamber->toGlobal((*segment).localPosition()), theSync);

        if(theGranularity == bySL) {
          vector<const TMax*> tMaxes = slSeg.getTMax(slId);
          DTWireId wireId(slId, 0, 0);
          theFile->cd();
          cellInfo* cell = theWireIdAndCellMap[wireId];
          if (cell==0) {
            TString name = (((((TString) "TMax"+(long) slId.wheel()) +(long) slId.station())
                             +(long) slId.sector())+(long) slId.superLayer());
            cell = new cellInfo(name);
            theWireIdAndCellMap[wireId] = cell;
          }
          cell->add(tMaxes);
          cell->update(); // FIXME to reset the counter to avoid triple counting, which actually is not used...
        }
        else {
          LogWarning("Calibration") << "[DTVDriftCalibration] ###Warning: the chosen granularity is not implemented yet, only bySL available!";
        }
        // to be implemented: granularity different from bySL

        //       else if (theGranularity == byPartition) {
        // 	// Use the custom granularity defined by partition(); 
        // 	// in this case, add() should be called once for each Tmax of each layer 
        // 	// and triple counting should be avoided within add()
        // 	vector<cellInfo*> cells;
        // 	for (int i=1; i<=4; i++) {
        // 	  const DTTMax::InfoLayer* iLayer = slSeg.getInfoLayer(i);
        // 	  if(iLayer == 0) continue;
        // 	  cellInfo * cell = partition(iLayer->idWire); 
        // 	  cells.push_back(cell);
        // 	  vector<const TMax*> tMaxes = slSeg.getTMax(iLayer->idWire);
        // 	  cell->add(tMaxes);
        // 	}
        // 	//reset the counter to avoid triple counting
        // 	for (vector<cellInfo*>::const_iterator i = cells.begin();
        // 	     i!= cells.end(); i++) {
        // 	  (*i)->update();
        // 	}
        //       } 
      }
    }
  }
}

void DTVDriftCalibration::endJob() {
  theFile->cd();
  gROOT->GetList()->Write();
  h2DSegmRPhi->Write();
  h2DSegmRZ->Write();
  h4DSegmAllCh->Write();
  hChi2->Write();
  // Instantiate a DTCalibrationMap object if you want to calculate the calibration constants
  DTCalibrationMap calibValuesFile(theCalibFilePar);  
  // Create the object to be written to DB
  DTMtime* mTime = new DTMtime();

  // write the TMax histograms of each SL to the root file
  if(theGranularity == bySL) {
    for(map<DTWireId, cellInfo*>::const_iterator  wireCell = theWireIdAndCellMap.begin();
        wireCell != theWireIdAndCellMap.end(); wireCell++) {
      cellInfo* cell= theWireIdAndCellMap[(*wireCell).first];
      hTMaxCell* cellHists = cell->getHists();
      theFile->cd();
      cellHists->Write();
      if(findVDriftAndT0) {  // if TRUE: evaluate calibration constants from TMax hists filled in this job  
	// evaluate v_drift and sigma from the TMax histograms
	DTWireId wireId = (*wireCell).first;
	vector<float> newConstants;
	TString N=(((((TString) "TMax"+(long) wireId.wheel()) +(long) wireId.station())
		    +(long) wireId.sector())+(long) wireId.superLayer());
	vector<float> vDriftAndReso = theFitter->evaluateVDriftAndReso(N);

	// Don't write the constants for the SL if the vdrift was not computed
	if(vDriftAndReso.front() == -1)
	  continue;
	const DTCalibrationMap::CalibConsts* oldConstants = calibValuesFile.getConsts(wireId);
	if(oldConstants != 0) {
	  newConstants.push_back((*oldConstants)[0]);
	  newConstants.push_back((*oldConstants)[1]);
	  newConstants.push_back((*oldConstants)[2]);
	} else {
	  newConstants.push_back(-1);
	  newConstants.push_back(-1);
	  newConstants.push_back(-1);
	}
	for(int ivd=0; ivd<=5;ivd++) { 
	  // 0=vdrift, 1=reso, 2=(3deltat0-2deltat0), 3=(2deltat0-1deltat0),
	  //  4=(1deltat0-0deltat0), 5=deltat0 from hists with max entries,
	  newConstants.push_back(vDriftAndReso[ivd]); 
	}

	calibValuesFile.addCell(calibValuesFile.getKey(wireId), newConstants);

        // vdrift is cm/ns , resolution is cm
	mTime->set((wireId.layerId()).superlayerId(),
		   vDriftAndReso[0],
		   vDriftAndReso[1],
		   DTVelocityUnits::cm_per_ns);
    	LogTrace("Calibration") << " SL: " << (wireId.layerId()).superlayerId()
	                        << " vDrift = " << vDriftAndReso[0]
	                        << " reso = " << vDriftAndReso[1];
      }
    }
  }

  // to be implemented: granularity different from bySL

  //   if(theGranularity == "byChamber") {
  //     const vector<DTChamber*> chambers = dMap.chambers();

  //     // Loop over all chambers
  //     for(vector<MuBarChamber*>::const_iterator chamber = chambers.begin();
  // 	chamber != chambers.end(); chamber ++) {
  //       MuBarChamberId chamber_id = (*chamber)->id();
  //       MuBarDigiParameters::Key wire_id(chamber_id, 0, 0, 0);
  //       vector<float> newConstants;
  //       vector<float> vDriftAndReso = evaluateVDriftAndReso(wire_id, f);
  //       const CalibConsts* oldConstants = digiParams.getConsts(wire_id);
  //       if(oldConstants !=0) {
  // 	newConstants = *oldConstants;
  // 	newConstants.push_back(vDriftAndReso[0]);
  // 	newConstants.push_back(vDriftAndReso[1]);
  // 	newConstants.push_back(vDriftAndReso[2]);
  // 	newConstants.push_back(vDriftAndReso[3]);
  //       } else {
  // 	newConstants.push_back(-1);
  // 	newConstants.push_back(-1);
  // 	newConstants.push_back(vDriftAndReso[0]);
  // 	newConstants.push_back(vDriftAndReso[1]);
  // 	newConstants.push_back(vDriftAndReso[2]);
  // 	newConstants.push_back(vDriftAndReso[3]);
  //       }
  //       digiParams.addCell(wire_id, newConstants);
  //     }
  //   }

  // Write values to a table  
  calibValuesFile.writeConsts(theVDriftOutputFile);

  LogVerbatim("Calibration") << "[DTVDriftCalibration]Writing vdrift object to DB!";

  // Write the vdrift object to DB
  string record = "DTMtimeRcd";
  DTCalibDBUtils::writeToDB<DTMtime>(record, mTime);
}

// to be implemented: granularity different from bySL

// // Create partitions 
// DTVDriftCalibration::cellInfo* DTVDriftCalibration::partition(const DTWireId& wireId) {
//   for( map<MuBarWireId, cellInfo*>::const_iterator iter =
// 	 mapCellTmaxPart.begin(); iter != mapCellTmaxPart.end(); iter++) {
//     // Divide wires per SL (with phi symmetry)
//     if(iter->first.wheel() == wireId.wheel() &&
//        iter->first.station() == wireId.station() &&
//        //       iter->first.sector() == wireId.sector() && // phi symmetry!
//        iter->first.superlayer() == wireId.superlayer()) {
//       return iter->second;
//     }
//   }
//   cellInfo * result = new cellInfo("dummy string"); // FIXME: change constructor; create tree?
//   mapCellTmaxPart.insert(make_pair(wireId, result));
//   return result;
//}


void DTVDriftCalibration::cellInfo::add(const vector<const TMax*>& _tMaxes) {
  vector<const TMax*> tMaxes = _tMaxes;
  float tmax123 = -1.;
  float tmax124 = -1.;
  float tmax134 = -1.;  
  float tmax234 = -1.;
  SigmaFactor s124 = noR;
  SigmaFactor s134 = noR;
  unsigned t0_123 = 0;
  unsigned t0_124 = 0;
  unsigned t0_134 = 0;
  unsigned t0_234 = 0;
  unsigned hSubGroup = 0;
  for (vector<const TMax*>::const_iterator it=tMaxes.begin(); it!=tMaxes.end();
       ++it) {
    if(*it == 0) {
      continue;  
    }
    else { 
      //FIXME check cached,
      if (addedCells.size()==4 || 
          find(addedCells.begin(), addedCells.end(), (*it)->cells) 
          != addedCells.end()) {
        continue;
      }
      addedCells.push_back((*it)->cells);    
      SigmaFactor sigma = (*it)->sigma;
      float t = (*it)->t;
      TMaxCells cells = (*it)->cells;
      unsigned t0Factor = (*it)->t0Factor;
      hSubGroup = (*it)->hSubGroup;
      if(t < 0.) continue;
      switch(cells) {
      case notInit : cout << "Error: no cell type assigned to TMax" << endl; break;
      case c123 : tmax123 =t; t0_123 = t0Factor; break;
      case c124 : tmax124 =t; s124 = sigma; t0_124 = t0Factor; break;
      case c134 : tmax134 =t; s134 = sigma; t0_134 = t0Factor; break;
      case c234 : tmax234 =t; t0_234 = t0Factor; break;
      } 
    }
  }
  //add entries to the TMax histograms
  histos->Fill(tmax123, tmax124, tmax134, tmax234, s124, s134, t0_123, 
               t0_124, t0_134, t0_234, hSubGroup); 
}

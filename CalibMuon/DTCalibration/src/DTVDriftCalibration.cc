
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/06/14 17:14:01 $
 *  $Revision: 1.1 $
 *  \author M. Giunta
 */

#include "DTVDriftCalibration.h"
#include "RecoLocalMuon/DTSegment/test/DTRecSegment4DReader.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "RecoLocalMuon/DTRecHit/interface/DTTTrigSyncFactory.h"
#include "RecoLocalMuon/DTRecHit/interface/DTTTrigBaseSync.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

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
extern void bookHistos();

using namespace std;
using namespace edm;
using namespace dttmaxenums;


DTVDriftCalibration::DTVDriftCalibration(const ParameterSet& pset) {
  bookHistos();

  debug = pset.getUntrackedParameter<bool>("debug","false");

  // the name of the 4D rec hits collection
  theRecHits4DLabel = pset.getUntrackedParameter<string>("recHits4DLabel");

  // the root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
  
  // the root file which will contain the histos
  theVDriftOutputFile = pset.getUntrackedParameter<string>("vDriftFileName");

  // Get the synchronizer
  theSync = DTTTrigSyncFactory::get()->create(pset.getUntrackedParameter<string>("tTrigMode"),
					      pset.getUntrackedParameter<ParameterSet>("tTrigModeConfig"));

  // get parameter set for DTCalibrationFile constructor
  theCalibFilePar =  pset.getUntrackedParameter<ParameterSet>("calibFileConfig");

  // the granularity to be used for tMax
  string tMaxGranularity = pset.getUntrackedParameter<string>("tMaxGranularity","bySL");
  
  // Initialize correctly the enum which specify the granularity for the calibration
  if(tMaxGranularity == "bySL") {
    theGranularity = bySL;
  } else if(tMaxGranularity == "byChamber"){
    theGranularity = byChamber;
  } else if(tMaxGranularity== "byPartition") {
    theGranularity = byPartition;
  } else {
    cout << "[DTVDriftCalibration]###Warning: Check parameter tMaxGranularity: "
	 << tMaxGranularity << " options not available!" << endl;
  }

  if(debug) 
    cout << "[DTVDriftCalibration]Constructor called!" << endl;
}

DTVDriftCalibration::~DTVDriftCalibration(){
  theFile->Close();
  if(debug) 
    cout << "[DTVDriftCalibration]Destructor called!" << endl;
}



void DTVDriftCalibration::analyze(const Event & event, const EventSetup& eventSetup) {
  cout << endl<<"--- [DTVDriftCalibration] Event analysed #Run: " << event.id().run()
       << " #Event: " << event.id().event() << endl;
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the rechit collection from the event
  Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel, all4DSegments);

  // Set the event setup in the Synchronizer 
  theSync->setES(eventSetup);

  // Loop over segments by chamber
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for (chamberIdIt = all4DSegments->id_begin();
       chamberIdIt != all4DSegments->id_end();
       ++chamberIdIt){

    // Get the chamber from the setup
    const DTChamber* chamber = dtGeom->chamber(*chamberIdIt);
    cout << "Chamber Id: " << *chamberIdIt << endl;

    // Get the range for the corresponding ChamberId
    DTRecSegment4DCollection::range  range = all4DSegments->get((*chamberIdIt));

    // Loop over the rechits of this DetUnit
    for (DTRecSegment4DCollection::const_iterator segment = range.first;
	 segment!=range.second; ++segment){
      cout << "Segment local pos (in chamber RF): " << (*segment).localPosition() << endl;
      cout << "Segment global pos: " << chamber->toGlobal((*segment).localPosition()) << endl;;

      const DTChamberRecSegment2D* phiSeg = (*segment).phiSegment();
      vector<DTRecHit1D> hits = phiSeg->specificRecHits();
      map<DTSuperLayerId,vector<DTRecHit1D> > hitsBySLMap; 
      for(vector<DTRecHit1D>::const_iterator hit = hits.begin();
	  hit != hits.end(); ++hit) {
	DTSuperLayerId slId =  (*hit).wireId().superlayerId();
	hitsBySLMap[slId].push_back(*hit); 
      }
      if((*segment).hasZed()) {
	const DTSLRecSegment2D* zSeg = (*segment).zSegment();
	hitsBySLMap[zSeg->superLayerId()] = zSeg->specificRecHits();
      }
      //loop over the segments 
      for(map<DTSuperLayerId,vector<DTRecHit1D> >::const_iterator slIdAndHits = hitsBySLMap.begin(); slIdAndHits != hitsBySLMap.end();  ++slIdAndHits) {
	if (slIdAndHits->second.size() < 3) continue;

	DTSuperLayerId slId =  slIdAndHits->first;
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
	  cout << "[DTVDriftCalibration]###Warning: the chosen granularity is not implemented yet, only bySL available!" << endl;
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
  cout << "End of job" << endl;
  theFile->cd(); 
  gROOT->GetList()->Write();
  // Instantiate a DTCalibrationFile object   
  DTCalibrationFile calibValuesFile(theCalibFilePar);  
  // write the TMax histograms of each SL to the root file
  if(theGranularity == bySL) {
    for(map<DTWireId, cellInfo*>::const_iterator  wireCell = theWireIdAndCellMap.begin();
	wireCell != theWireIdAndCellMap.end(); wireCell++) {
      cellInfo* cell= theWireIdAndCellMap[(*wireCell).first];
      hTMaxCell* cellHists = cell->getHists();
      cellHists->Write();
      
      // evaluate v_drift and sigma from the TMax histograms
      DTWireId wireId = (*wireCell).first;
      vector<float> newConstants;
      vector<float> vDriftAndReso = evaluateVDriftAndReso(wireId);
      // Don't write the constants for the SL if the vdrift was not computed
      if(vDriftAndReso.front() == -1)
	continue;

      const DTCalibrationFile::CalibConsts* oldConstants = calibValuesFile.getConsts(wireId);
      if(oldConstants != 0) {
	newConstants.push_back((*oldConstants)[0]);
	newConstants.push_back((*oldConstants)[1]);
	newConstants.push_back(vDriftAndReso[0]); // vdrift
	newConstants.push_back(vDriftAndReso[1]); // reso
	newConstants.push_back(vDriftAndReso[2]); // delta(t0) calculated from gaussian mean
	newConstants.push_back(vDriftAndReso[3]); // delta(t0) calculated from histo mean
	
      } else {
	newConstants.push_back(-1);
	newConstants.push_back(-1);
	newConstants.push_back(vDriftAndReso[0]);
	newConstants.push_back(vDriftAndReso[1]);
	newConstants.push_back(vDriftAndReso[2]);
	newConstants.push_back(vDriftAndReso[3]);
      }
      calibValuesFile.addCell(calibValuesFile.getKey(wireId), newConstants);
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
  //write values to a table  
  calibValuesFile.writeConsts(theVDriftOutputFile);
}

vector<float> DTVDriftCalibration::evaluateVDriftAndReso (const DTWireId& wireId) {
  TString N=(((((TString) "TMax"+(long) wireId.wheel()) +(long) wireId.station())
	      +(long) wireId.sector())+(long) wireId.superLayer());
  cout << N << endl;
  // Retrieve histogram sets
  hTMaxCell * histos   = new hTMaxCell(N, theFile);
  vector<float> vDriftAndReso;

  // Check that the histo for this cell exists
  if(histos->hTmax123 != 0) {
    vector<TH1F*> hTMax;  // histograms for <T_max> calculation
    vector <TH1F*> hT0;   // histograms for T0 evaluation
    hTMax.push_back(histos->hTmax123); 
    hTMax.push_back(histos->hTmax124s72);
    hTMax.push_back(histos->hTmax124s78);
    hTMax.push_back(histos->hTmax134s72);
    hTMax.push_back(histos->hTmax134s78);
    hTMax.push_back(histos->hTmax234);

    hT0.push_back(histos->hTmax_3t0);
    hT0.push_back(histos->hTmax_2t0);
    hT0.push_back(histos->hTmax_t0);
    hT0.push_back(histos->hTmax_0);

    vector<Double_t> factor; // factor relating the width of the Tmax distribution 
                             // and the cell resolution 
    factor.push_back(sqrt(2./3.)); // hTmax123
    factor.push_back(sqrt(2./7.)); // hTmax124s72
    factor.push_back(sqrt(8./7.)); // hTmax124s78
    factor.push_back(sqrt(2./7.)); // hTmax134s72
    factor.push_back(sqrt(8./7.)); // hTmax134s78
    factor.push_back(sqrt(2./3.)); // hTmax234


    // Retrieve the gaussian mean and sigma for each histogram    
    vector<Double_t> mean;
    vector<Double_t> sigma; 
    vector<Double_t> count;  //number of entries

    for(vector<TH1F*>::const_iterator ith = hTMax.begin();
	ith != hTMax.end(); ith++) {
      // Find distribution peak and fit range
      Int_t peak = (*ith)->GetMaximumBin() - 1000;
      Double_t range = 2.*(*ith)->GetRMS(); 

      // Fit each Tmax histogram with a Gaussian in a restricted interval
      TF1 *rGaus = new TF1("rGaus","gaus",peak-range,peak+range);
      (*ith)->Fit("rGaus","R");
      TF1 *f1 = (*ith)->GetFunction("rGaus");
      // Get mean, sigma and number of entries of each histogram
      mean.push_back(f1->GetParameter(1));
      sigma.push_back(f1->GetParameter(2)); 
      count.push_back((*ith)->GetEntries());  
    } 
  	  
    Double_t tMaxMean=0.;
    Double_t wTMaxSum=0.;
    Double_t sigmaT=0.;
    Double_t wSigmaSum = 0.;
  
    //calculate total mean and sigma
    for(int i=0; i<=5; i++) {
      if(count[i]<200) continue;
      tMaxMean  += mean[i]*(count[i]/(sigma[i]*sigma[i]));
      wTMaxSum  += count[i]/(sigma[i]*sigma[i]);
      sigmaT    += count[i]*factor[i]*sigma[i];
      wSigmaSum += count[i];
      // cout << "TMaxMean "<<i<<": "<< mean[i] << " entries: " << count[i] 
      // << " sigma: " << sigma[i] 
      // << " weight: " << (count[i]/(sigma[i]*sigma[i])) << endl; 
    }
    tMaxMean /= wTMaxSum;
    sigmaT /= wSigmaSum;

    //calculate v_drift and resolution
    Double_t vDrift = 2.1 / tMaxMean; //2.1 is the half cell length in cm
    Double_t reso = vDrift * sigmaT;
    vDriftAndReso.push_back(vDrift);
    vDriftAndReso.push_back(reso);
    cout << " final TMaxMean=" << tMaxMean << " sigma= "  << sigmaT 
	 << " v_d and reso: " << vDrift << " " << reso << endl;

    //Retrieve t0 histogram number of entries (choose histograms with higher nr. of entries)
    map<Double_t,TH1F*> hEntries;

    for(vector<TH1F*>::const_iterator ith = hT0.begin();
	ith != hT0.end(); ith++) {
      hEntries[(*ith)->GetEntries()] = (*ith);
      
    } 

    vector<TH1F*> t0Hists;
    int counter = 0;
    
    for(map<Double_t,TH1F*>::reverse_iterator iter = hEntries.rbegin();
	iter != hEntries.rend(); iter++) {
      counter++;
      cout << iter->first << " " << iter->second->GetTitle() << endl;
      if (counter==1) t0Hists.push_back(iter->second); //first hist selected for t0 evaluation
      else if (counter==2) {t0Hists.push_back(iter->second); break;} //second hist selected for t0 evaluation
    }
      
    // Retrieve the gaussian mean and sigma of histograms for Delta(t0) evaluation   
    vector<Double_t> meanT0;
    vector<Double_t> sigmaT0; 
    vector<Double_t> countT0; 

    for(vector<TH1F*>::const_iterator ith = t0Hists.begin();
	ith != t0Hists.end(); ith++) {
      cout << (*ith)->GetTitle() << endl;
      (*ith)->Fit("gaus");
      TF1 *f1 = (*ith)->GetFunction("gaus");
      // Get mean, sigma and number of entries of each the two selected histograms
      meanT0.push_back(f1->GetParameter(1));
      sigmaT0.push_back(f1->GetParameter(2));
      countT0.push_back((*ith)->GetEntries());
    }
    
    //calculate Delta(t0)
    Double_t t0Diff = histos->GetT0Factor(t0Hists[0]) - 
      histos->GetT0Factor(t0Hists[1]);
    
    Double_t deltaT0_fromHMean = (t0Hists[0]->GetMean(1) - 
				  t0Hists[1]->GetMean(1)) / t0Diff; 
    Double_t deltaT0 = -1;
    if((countT0[0] > 200) && (countT0[1] > 200)) 
      deltaT0 = (meanT0[0] - meanT0[1]) / t0Diff; 
    
    vDriftAndReso.push_back(deltaT0);
    vDriftAndReso.push_back(deltaT0_fromHMean);
  } else {
    vDriftAndReso.push_back(-1);
    vDriftAndReso.push_back(-1);
    vDriftAndReso.push_back(-1);
    vDriftAndReso.push_back(-1);
  }
  
  return vDriftAndReso;

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
}


void DTVDriftCalibration::cellInfo::add(vector<const TMax*> tMaxes) {
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
  unsigned hSubGroup;
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

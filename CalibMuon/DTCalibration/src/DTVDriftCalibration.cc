
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/06/22 17:40:55 $
 *  $Revision: 1.2 $
 *  \author M. Giunta
 */

#include "CalibMuon/DTCalibration/src/DTVDriftCalibration.h"
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
TH1F * hChi2;
extern void bookHistos();

using namespace std;
using namespace edm;
using namespace dttmaxenums;


DTVDriftCalibration::DTVDriftCalibration(const ParameterSet& pset) {
  hChi2 = new TH1F("hChi2","Chi squared tracks",100,0,100);
  h2DSegmRPhi = new h2DSegm("SLRPhi");
  h2DSegmRZ = new h2DSegm("SLRZ");
  bookHistos();

  debug = pset.getUntrackedParameter<bool>("debug", "false");

  findVDriftAndT0 = pset.getUntrackedParameter<bool>("findVDriftAndT0", "false");

  // Chamber/s to calibrate
  theCalibChamber =  pset.getUntrackedParameter<string>("calibChamber", "All");
 
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

  // get parameter set for DTCalibrationMap constructor
  theCalibFilePar =  pset.getUntrackedParameter<ParameterSet>("calibFileConfig");

  // get maximum chi2 value 
  theMaxChi2 =  pset.getParameter<double>("maxChi2");

  // Maximum incidence angle for Phi SL 
  theMaxPhiAngle =  pset.getParameter<double>("maxAnglePhi");

  // Maximum incidence angle for Theta SL 
  theMaxZAngle =  pset.getParameter<double>("maxAngleZ");
  
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

  DTChamberId chosenChamberId;
  if(theCalibChamber != "All") {
    stringstream linestr;
    int selWheel, selStation, selSector;
    linestr << theCalibChamber;
    linestr >> selWheel >> selStation >> selSector;
    chosenChamberId = DTChamberId(selWheel, selStation, selSector);
    cout << "chosen chamber " << chosenChamberId << endl;
  }
 
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
    if(debug)
      cout << "Chamber Id: " << *chamberIdIt << endl;


    // Calibrate just the chosen chamber/s    
    if((theCalibChamber != "All") && ((*chamberIdIt) != chosenChamberId)) 
      continue;
    
    // Get the range for the corresponding ChamberId
    DTRecSegment4DCollection::range  range = all4DSegments->get((*chamberIdIt));
    
    // Loop over the rechits of this DetUnit
    for (DTRecSegment4DCollection::const_iterator segment = range.first;
	 segment!=range.second; ++segment){
      if(debug) {
	cout << "Segment local pos (in chamber RF): " << (*segment).localPosition() << endl;
	cout << "Segment global pos: " << chamber->toGlobal((*segment).localPosition()) << endl;;
      }

      //get the segment chi2
      double chiSquare = ((*segment).chi2()/(*segment).degreesOfFreedom());
      hChi2->Fill(chiSquare);
      // cut on the segment chi2 
      if(chiSquare > theMaxChi2) continue;

     // get the Phi 2D segment and plot the angle in the chamber RF
      const DTChamberRecSegment2D* phiSeg = (*segment).phiSegment();  // phiSeg lives in the chamber RF
      LocalPoint phiSeg2DPosInCham = phiSeg->localPosition();  
      LocalVector phiSeg2DDirInCham = phiSeg->localDirection();

      if(fabs(atan(phiSeg2DDirInCham.x()/phiSeg2DDirInCham.z())) > theMaxPhiAngle) continue; // cut on the angle

      h2DSegmRPhi->Fill(phiSeg2DPosInCham.x(), phiSeg2DDirInCham.x()/phiSeg2DDirInCham.z());
      
      vector<DTRecHit1D> hits = phiSeg->specificRecHits();
      map<DTSuperLayerId,vector<DTRecHit1D> > hitsBySLMap; 
      for(vector<DTRecHit1D>::const_iterator hit = hits.begin();
	  hit != hits.end(); ++hit) {
	DTSuperLayerId slId =  (*hit).wireId().superlayerId();
	hitsBySLMap[slId].push_back(*hit); 
      }
     // get the Theta 2D segment and plot the angle in the chamber RF
      if((*segment).hasZed()) {
	const DTSLRecSegment2D* zSeg = (*segment).zSegment();  // zSeg lives in the SL RF
	const DTSuperLayer* sl = chamber->superLayer(zSeg->superLayerId());
	LocalPoint zSeg2DPosInCham = chamber->toLocal(sl->toGlobal((*zSeg).localPosition())); 
	LocalVector zSeg2DDirInCham = chamber->toLocal(sl->toGlobal((*zSeg).localDirection()));

	if(fabs(atan(zSeg2DDirInCham.y()/zSeg2DDirInCham.z())) > theMaxZAngle) continue; // cut on the angle

	h2DSegmRZ->Fill(zSeg2DPosInCham.y(), zSeg2DDirInCham.y()/zSeg2DDirInCham.z());
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
  theFile->cd();
  gROOT->GetList()->Write();
  h2DSegmRPhi->Write();
  h2DSegmRZ->Write();
  hChi2->Write();
  // Instantiate a DTCalibrationMap object if you want to calculate the calibration constants
  DTCalibrationMap calibValuesFile(theCalibFilePar);  
  // write the TMax histograms of each SL to the root file
  if(theGranularity == bySL) {
    for(map<DTWireId, cellInfo*>::const_iterator  wireCell = theWireIdAndCellMap.begin();
	wireCell != theWireIdAndCellMap.end(); wireCell++) {
      cellInfo* cell= theWireIdAndCellMap[(*wireCell).first];
      hTMaxCell* cellHists = cell->getHists();
      cellHists->Write();
      if(findVDriftAndT0) {  // if TRUE: evaluate calibration constants from TMax hists filled in this job  
	// evaluate v_drift and sigma from the TMax histograms
	DTWireId wireId = (*wireCell).first;
	vector<float> newConstants;
	vector<float> vDriftAndReso = evaluateVDriftAndReso(wireId);

	// Don't write the constants for the SL if the vdrift was not computed
	if(vDriftAndReso.front() == -1)
	  continue;
	const DTCalibrationMap::CalibConsts* oldConstants = calibValuesFile.getConsts(wireId);
	if(oldConstants != 0) {
	  newConstants.push_back((*oldConstants)[0]);
	  newConstants.push_back((*oldConstants)[1]);
	} else {
	  newConstants.push_back(-1);
	  newConstants.push_back(-1);
	}
	for(int ivd=0; ivd<=5;ivd++) { 
	  // 0=vdrift, 1=reso, 2=(3deltat0-2deltat0), 3=(2deltat0-1deltat0),
	  //  4=(1deltat0-0deltat0), 5=deltat0 from hists with max entries,
	  newConstants.push_back(vDriftAndReso[ivd]); 
	}
	calibValuesFile.addCell(calibValuesFile.getKey(wireId), newConstants);
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

  //write values to a table  
  calibValuesFile.writeConsts(theVDriftOutputFile);
}

vector<float> DTVDriftCalibration::evaluateVDriftAndReso (const DTWireId& wireId) {
  TString N=(((((TString) "TMax"+(long) wireId.wheel()) +(long) wireId.station())
	      +(long) wireId.sector())+(long) wireId.superLayer());
  
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


    // Retrieve the gaussian mean and sigma for each TMax histogram    
    vector<Double_t> mean;
    vector<Double_t> sigma; 
    vector<Double_t> count;  //number of entries

    for(vector<TH1F*>::const_iterator ith = hTMax.begin();
	ith != hTMax.end(); ith++) {
      // Find distribution peak and fit range
      Double_t peak = ((((((*ith)->GetXaxis())->GetXmax())-(((*ith)->GetXaxis())->GetXmin()))/(*ith)->GetNbinsX())*
		       ((*ith)->GetMaximumBin()))+(((*ith)->GetXaxis())->GetXmin());
      if(debug)
	cout<<"Peak "<<peak<<" : "<<"xmax "<<(((*ith)->GetXaxis())->GetXmax())
	    <<"            xmin "<<(((*ith)->GetXaxis())->GetXmin())
	    <<"            nbin "<<(*ith)->GetNbinsX()
	    <<"            bin with max "<<((*ith)->GetMaximumBin())<<endl;
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
    if(debug)
      cout << " final TMaxMean=" << tMaxMean << " sigma= "  << sigmaT 
	   << " v_d and reso: " << vDrift << " " << reso << endl;

    // Order t0 histogram by number of entries (choose histograms with higher nr. of entries)
    map<Double_t,TH1F*> hEntries;    
    for(vector<TH1F*>::const_iterator ith = hT0.begin();
	ith != hT0.end(); ith++) {
      hEntries[(*ith)->GetEntries()] = (*ith);
    } 

    // add at the end of hT0 the two hists with the higher number of entries 
    int counter = 0;
    for(map<Double_t,TH1F*>::reverse_iterator iter = hEntries.rbegin();
 	iter != hEntries.rend(); iter++) {
      counter++;
      if (counter==1) hT0.push_back(iter->second); 
      else if (counter==2) {hT0.push_back(iter->second); break;} 
    }
    
    // Retrieve the gaussian mean and sigma of histograms for Delta(t0) evaluation   
    vector<Double_t> meanT0;
    vector<Double_t> sigmaT0; 
    vector<Double_t> countT0;

    for(vector<TH1F*>::const_iterator ith = hT0.begin();
	ith != hT0.end(); ith++) {
      (*ith)->Fit("gaus");
      TF1 *f1 = (*ith)->GetFunction("gaus");
      // Get mean, sigma and number of entries of the  histograms
      meanT0.push_back(f1->GetParameter(1));
      sigmaT0.push_back(f1->GetParameter(2));
      countT0.push_back((*ith)->GetEntries());
    }
    //calculate Delta(t0)
    if(hT0.size() != 6) { // check if you have all the t0 hists
      cout << "t0 histograms = " << hT0.size() << endl;
      for(int i=1; i<=4;i++) {
	vDriftAndReso.push_back(-1);
      }
      return vDriftAndReso;
    }
    
    for(int it0=0; it0<=2; it0++) {      
      if((countT0[it0] > 200) && (countT0[it0+1] > 200)) {
	Double_t deltaT0 = meanT0[it0] - meanT0[it0+1];	
	vDriftAndReso.push_back(deltaT0);
      }  
      else
 	vDriftAndReso.push_back(999.);
    }
    //deltat0 using hists with max nr. of entries
    if((countT0[4] > 200) && (countT0[5] > 200)) {
      Double_t t0Diff = histos->GetT0Factor(hT0[4]) - histos->GetT0Factor(hT0[5]);
      Double_t deltaT0MaxEntries =  (meanT0[4] - meanT0[5])/ t0Diff;
      vDriftAndReso.push_back(deltaT0MaxEntries);
    }
    else
      vDriftAndReso.push_back(999.);
  }
  else {
    for(int i=1; i<=6; i++) { 
      // 0=vdrift, 1=reso,  2=(3deltat0-2deltat0), 3=(2deltat0-1deltat0), 
      // 4=(1deltat0-0deltat0), 5=deltat0 from hists with max entries,
      vDriftAndReso.push_back(-1);
    }
  }
  return vDriftAndReso;
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

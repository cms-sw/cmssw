
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/08/31 15:30:32 $
 *  $Revision: 1.1 $
 *  \author M. Giunta
 */

#include "CalibMuon/DTCalibration/src/DTVDriftWriter.h"
#include "CalibMuon/DTCalibration/src/vDriftHistos.h"
#include "CalibMuon/DTCalibration/src/DTCalibrationMap.h"
#include "RecoLocalMuon/DTSegment/test/DTRecSegment4DReader.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
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
#include "TFile.h"

using namespace std;
using namespace edm;
//using namespace dttmaxenums;


DTVDriftWriter::DTVDriftWriter(const ParameterSet& pset) {
  // get selected debug option
  debug = pset.getUntrackedParameter<bool>("debug", "false");

  // Open the root file which contains the histos
  theRootInputFile = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(theRootInputFile.c_str(), "READ");
  
  // the text file which will contain the histos
  theVDriftOutputFile = pset.getUntrackedParameter<string>("vDriftFileName");

  // get parameter set for DTCalibrationMap constructor
  theCalibFilePar =  pset.getUntrackedParameter<ParameterSet>("calibFileConfig");

  // the granularity to be used for calib constants evaluation
  theGranularity = pset.getUntrackedParameter<string>("calibGranularity","bySL");
  
  //tag for the DB
  string tag = pset.getUntrackedParameter<string>("meanTimerTag", "vDrift");
  theMTime = new DTMtime(tag);

  if(debug)
    cout << "[DTVDriftWriter]Constructor called!" << endl;
}

DTVDriftWriter::~DTVDriftWriter(){
  if(debug)
    cout << "[DTVDriftWriter]Destructor called!" << endl;
}

void DTVDriftWriter::analyze(const Event & event, const EventSetup& eventSetup) {
  if(debug)
    cout << "[DTVDriftWriter]Analyzer called!" << endl;

  // Instantiate a DTCalibrationMap object 
  DTCalibrationMap calibValuesFile(theCalibFilePar);  

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  if(theGranularity == "bySL") {    
    // Get all the sls from the setup
    const vector<DTSuperLayer*> superLayers = dtGeom->superLayers(); 
    
    // Loop over all SLs
    for(vector<DTSuperLayer*>::const_iterator  slCell = superLayers.begin();
	slCell != superLayers.end(); slCell++) {
      
      DTSuperLayerId slId = (*slCell)->id();
      // evaluate v_drift and sigma from the TMax histograms
      DTWireId wireId(slId, 0, 0);
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

      theMTime->setSLMtime(slId,
			   vDriftAndReso[0],
			   vDriftAndReso[1],
			   DTTimeUnits::ns);
      if(debug) {
	cout << " SL: " << slId
	     << " vDrift = " << vDriftAndReso[0]
	     << " reso = " << vDriftAndReso[1] << endl;
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

void DTVDriftWriter::endJob() {

if(debug) 
   cout << "[DTVDriftWriter]Writing vdrift object to DB!" << endl;

  // Write the ttrig object to DB
  edm::Service<cond::service::PoolDBOutputService> dbOutputSvc;
  if( dbOutputSvc.isAvailable() ){
    size_t callbackToken = dbOutputSvc->callbackToken("DTDBObject");
    try{
      dbOutputSvc->newValidityForNewPayload<DTMtime>(theMTime, dbOutputSvc->endOfTime(), callbackToken);
    }catch(const cond::Exception& er){
      cout << er.what() << endl;
    }catch(const std::exception& er){
      cout << "[DTVDriftWriter] caught std::exception " << er.what() << endl;
    }catch(...){
      cout << "[DTVDriftWriter] Funny error" << endl;
    }
  }else{
    cout << "Service PoolDBOutputService is unavailable" << endl;
  }

}

vector<float> DTVDriftWriter::evaluateVDriftAndReso (const DTWireId& wireId) {
  TString N=(((((TString) "TMax"+(long) wireId.wheel()) +(long) wireId.station())
	      +(long) wireId.sector())+(long) wireId.superLayer());
  cout << "[evaluateVDriftAndReso] called for wire: " << wireId << endl; 
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
      //cout << "TMaxMean "<<i<<": "<< mean[i] << " entries: " << count[i] 
      //   << " sigma: " << sigma[i] 
      //   << " weight: " << (count[i]/(sigma[i]*sigma[i])) << endl; 
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
  // DTVDriftWriter::cellInfo* DTVDriftWriter::partition(const DTWireId& wireId) {
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

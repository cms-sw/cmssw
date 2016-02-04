/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/09/21 08:03:43 $
 *  $Revision: 1.6 $
 *  \author G. Cerminara - INFN Torino
 */
#include "CalibMuon/DTCalibration/plugins/DTTTrigCalibration.h"
#include "CalibMuon/DTCalibration/interface/DTTimeBoxFitter.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"


#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"


#include "TFile.h"
#include "TH1F.h"
#include "TGraph.h"

class DTLayer;

using namespace std;
using namespace edm;
// using namespace cond;



// Constructor
DTTTrigCalibration::DTTTrigCalibration(const edm::ParameterSet& pset) {
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");

  // Get the label to retrieve digis from the event
  digiLabel = pset.getUntrackedParameter<string>("digiLabel");

  // Switch on/off the DB writing
  findTMeanAndSigma = pset.getUntrackedParameter<bool>("fitAndWrite", false);

  // The TDC time-window (ns)
  maxTDCCounts = 5000 * pset.getUntrackedParameter<int>("tdcRescale", 1);
  //The maximum number of digis per layer
  maxDigiPerLayer = pset.getUntrackedParameter<int>("maxDigiPerLayer", 10);

  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
  theFitter = new DTTimeBoxFitter();
  if(debug)
    theFitter->setVerbosity(1);
  
  double sigmaFit = pset.getUntrackedParameter<double>("sigmaTTrigFit",10.);
  theFitter->setFitSigma(sigmaFit);

  doSubtractT0 = pset.getUntrackedParameter<bool>("doSubtractT0","false");
  // Get the synchronizer
  if(doSubtractT0) {
    theSync = DTTTrigSyncFactory::get()->create(pset.getUntrackedParameter<string>("tTrigMode"),
						pset.getUntrackedParameter<ParameterSet>("tTrigModeConfig"));
  } else {
    theSync = 0;
  }

  checkNoisyChannels = pset.getUntrackedParameter<bool>("checkNoisyChannels","false");

  // the kfactor to be uploaded in the ttrig DB
  kFactor =  pset.getUntrackedParameter<double>("kFactor",-0.7);

  if(debug) 
    cout << "[DTTTrigCalibration]Constructor called!" << endl;
}



// Destructor
DTTTrigCalibration::~DTTTrigCalibration(){
  if(debug) 
    cout << "[DTTTrigCalibration]Destructor called!" << endl;

  //   // Delete all histos
  //   for(map<DTSuperLayerId, TH1F*>::const_iterator slHisto = theHistoMap.begin();
  //       slHisto != theHistoMap.end();
  //       slHisto++) {
  //     delete (*slHisto).second;
  //   }

  theFile->Close();
  delete theFitter;
}



/// Perform the real analysis
void DTTTrigCalibration::analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {

  if(debug) 
    cout << "[DTTTrigCalibration] #Event: " << event.id().event() << endl;
  
  // Get the digis from the event
  Handle<DTDigiCollection> digis; 
  event.getByLabel(digiLabel, digis);

  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    // Get the map of noisy channels
    eventSetup.get<DTStatusFlagRcd>().get(statusMap);
  }

  if(doSubtractT0)
    theSync->setES(eventSetup);
 
  //The chambers too noisy in this event
  vector<DTChamberId> badChambers;

  // Iterate through all digi collections ordered by LayerId   
  DTDigiCollection::DigiRangeIterator dtLayerIt;
  for (dtLayerIt = digis->begin();
       dtLayerIt != digis->end();
       ++dtLayerIt){
    // Get the iterators over the digis associated with this LayerId
    const DTDigiCollection::Range& digiRange = (*dtLayerIt).second; 
    
    const DTLayerId layerId = (*dtLayerIt).first;
    const DTSuperLayerId slId = layerId.superlayerId();
    const DTChamberId chId = slId.chamberId();
    bool badChamber=false;

    if(debug)
      cout<<"----------- Layer "<<layerId<<" -------------"<<endl;

    //Check if the layer is inside a noisy chamber
    for(vector<DTChamberId>::const_iterator chamber = badChambers.begin(); chamber != badChambers.end(); chamber++){
      if((*chamber) == chId){
	badChamber=true;
	break;
      }
    }
    if(badChamber) continue;

    //Check if the layer has too many digis
    if((digiRange.second - digiRange.first) > maxDigiPerLayer){
      if(debug)
	cout<<"Layer "<<layerId<<"has too many digis ("<<(digiRange.second - digiRange.first)<<")"<<endl;
      badChambers.push_back(chId);
      continue;
    }

    // Get the histo from the map
    TH1F *hTBox = theHistoMap[slId];
    if(hTBox == 0) {
      // Book the histogram
      theFile->cd();
      hTBox = new TH1F(getTBoxName(slId).c_str(), "Time box (ns)", int(0.25*32.0*maxTDCCounts/25.0), 0, maxTDCCounts);
      if(debug)
	cout << "  New Time Box: " << hTBox->GetName() << endl;
      theHistoMap[slId] = hTBox;
    }
    TH1F *hO = theOccupancyMap[layerId];
    if(hO == 0) {
      // Book the histogram
      theFile->cd();
      hO = new TH1F(getOccupancyName(layerId).c_str(), "Occupancy", 100, 0, 100);
      if(debug)
	cout << "  New Time Box: " << hO->GetName() << endl;
      theOccupancyMap[layerId] = hO;
    }

    // Loop over all digis in the given range
    for (DTDigiCollection::const_iterator digi = digiRange.first;
	 digi != digiRange.second;
	 digi++) {
      const DTWireId wireId(layerId, (*digi).wire());

      // Check for noisy channels and skip them
      if(checkNoisyChannels) {
	bool isNoisy = false;
	bool isFEMasked = false;
	bool isTDCMasked = false;
	bool isTrigMask = false;
	bool isDead = false;
	bool isNohv = false;
	statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
	if(isNoisy) {
	  if(debug)
	    cout << "Wire: " << wireId << " is noisy, skipping!" << endl;
	  continue;
	}      
      }
      theFile->cd();
      double offset = 0;
      if(doSubtractT0) {
	const DTLayer* layer = 0;//fake
	const GlobalPoint glPt;//fake
	offset = theSync->offset(layer, wireId, glPt);
      }
      hTBox->Fill((*digi).time()-offset);
      if(debug) {
	cout << "   Filling Time Box:   " << hTBox->GetName() << endl;
	cout << "           offset (ns): " << offset << endl;
	cout << "           time(ns):   " << (*digi).time()-offset<< endl;
      }
      hO->Fill((*digi).wire());
    }
  }
}


void DTTTrigCalibration::endJob() {
  if(debug) 
    cout << "[DTTTrigCalibration]Writing histos to file!" << endl;
  
  // Write all time boxes to file
  theFile->cd();
  for(map<DTSuperLayerId, TH1F*>::const_iterator slHisto = theHistoMap.begin();
      slHisto != theHistoMap.end();
      slHisto++) {
    (*slHisto).second->Write();
  }
  for(map<DTLayerId, TH1F*>::const_iterator slHisto = theOccupancyMap.begin();
      slHisto != theOccupancyMap.end();
      slHisto++) {
    (*slHisto).second->Write();
  }

  if(findTMeanAndSigma) {
      // Create the object to be written to DB
      DTTtrig* tTrig = new DTTtrig();

      // Loop over the map, fit the histos and write the resulting values to the DB
      for(map<DTSuperLayerId, TH1F*>::const_iterator slHisto = theHistoMap.begin();
	  slHisto != theHistoMap.end();
	  slHisto++) {
	pair<double, double> meanAndSigma = theFitter->fitTimeBox((*slHisto).second);
	tTrig->set((*slHisto).first,
		   meanAndSigma.first,
		   meanAndSigma.second,
                   kFactor,
		   DTTimeUnits::ns);
    
	if(debug) {
	  cout << " SL: " << (*slHisto).first
	       << " mean = " << meanAndSigma.first
	       << " sigma = " << meanAndSigma.second << endl;
	}
      }

      // Print the ttrig map
      dumpTTrigMap(tTrig);
  
      // Plot the tTrig
      plotTTrig(tTrig);

      if(debug) 
	cout << "[DTTTrigCalibration]Writing ttrig object to DB!" << endl;


      // FIXME: to be read from cfg?
      string tTrigRecord = "DTTtrigRcd";
      
      // Write the object to DB
      DTCalibDBUtils::writeToDB(tTrigRecord, tTrig);
    }  

}




string DTTTrigCalibration::getTBoxName(const DTSuperLayerId& slId) const {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" << slId.wheel() << "_" << slId.station() << "_" << slId.sector()
	    << "_SL" << slId.superlayer() << "_hTimeBox";
  theStream >> histoName;
  return histoName;
}

string DTTTrigCalibration::getOccupancyName(const DTLayerId& slId) const {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" << slId.wheel() << "_" << slId.station() << "_" << slId.sector()
	    << "_SL" << slId.superlayer() << "_L"<< slId.layer() <<"_Occupancy";
  theStream >> histoName;
  return histoName;
}


void DTTTrigCalibration::dumpTTrigMap(const DTTtrig* tTrig) const {
  static const double convToNs = 25./32.;
  for(DTTtrig::const_iterator ttrig = tTrig->begin();
      ttrig != tTrig->end(); ttrig++) {
    cout << "Wh: " << (*ttrig).first.wheelId
	 << " St: " << (*ttrig).first.stationId
	 << " Sc: " << (*ttrig).first.sectorId
	 << " Sl: " << (*ttrig).first.slId
	 << " TTrig mean (ns): " << (*ttrig).second.tTrig * convToNs
	 << " TTrig sigma (ns): " << (*ttrig).second.tTrms * convToNs<< endl;
  }
}


void DTTTrigCalibration::plotTTrig(const DTTtrig* tTrig) const {

  TH1F* tTrig_YB1_Se10 = new TH1F("tTrig_YB1_Se10","tTrig YB1_Se10",15,1,16);
  TH1F* tTrig_YB2_Se10 = new TH1F("tTrig_YB2_Se10","tTrig YB2_Se10",15,1,16);
  TH1F* tTrig_YB2_Se11 = new TH1F("tTrig_YB2_Se11","tTrig YB2_Se11",12,1,13);

  static const double convToNs = 25./32.;
  for(DTTtrig::const_iterator ttrig = tTrig->begin();
      ttrig != tTrig->end(); ttrig++) {

    // avoid to have wired numbers in the plot
    float tTrigValue=0;
    float tTrmsValue=0;
    if ((*ttrig).second.tTrig * convToNs > 0 &&
	(*ttrig).second.tTrig * convToNs < 32000 ) {
      tTrigValue = (*ttrig).second.tTrig * convToNs;
      tTrmsValue = (*ttrig).second.tTrms * convToNs;
    }

    int binx;
    string binLabel;
    stringstream binLabelStream;
    if ((*ttrig).first.sectorId != 14) {
      binx = ((*ttrig).first.stationId-1)*3  + (*ttrig).first.slId;
      binLabelStream << "MB"<<(*ttrig).first.stationId<<"_SL"<<(*ttrig).first.slId;
    }
    else {
      binx = 12  + (*ttrig).first.slId;
      binLabelStream << "MB14_SL"<<(*ttrig).first.slId;
    }
    binLabelStream >> binLabel;

    if ((*ttrig).first.wheelId == 2) {
      if ((*ttrig).first.sectorId == 10 || (*ttrig).first.sectorId == 14) {
	tTrig_YB2_Se10->Fill( binx,tTrigValue);
	tTrig_YB2_Se10->SetBinError( binx, tTrmsValue);
	tTrig_YB2_Se10->GetXaxis()->SetBinLabel(binx,binLabel.c_str());
	tTrig_YB2_Se10->GetYaxis()->SetTitle("ns");
      }
      else {
	tTrig_YB2_Se11->Fill( binx,tTrigValue);
	tTrig_YB2_Se11->SetBinError( binx,tTrmsValue);
	tTrig_YB2_Se11->GetXaxis()->SetBinLabel(binx,binLabel.c_str());
	tTrig_YB2_Se11->GetYaxis()->SetTitle("ns");
      }
    }
    else {
      tTrig_YB1_Se10->Fill( binx,tTrigValue);
      tTrig_YB1_Se10->SetBinError( binx,tTrmsValue);
      tTrig_YB1_Se10->GetXaxis()->SetBinLabel(binx,binLabel.c_str());
      tTrig_YB1_Se10->GetYaxis()->SetTitle("ns");
    }
  }

  tTrig_YB1_Se10->Write();
  tTrig_YB2_Se10->Write();
  tTrig_YB2_Se11->Write();

}

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/06/15 13:47:33 $
 *  $Revision: 1.13 $
 *  \author G. Cerminara - INFN Torino
 */
#include "CalibMuon/DTCalibration/src/DTTTrigCalibration.h"
#include "CalibMuon/DTCalibration/interface/DTTimeBoxFitter.h"
#include "RecoLocalMuon/DTRecHit/interface/DTTTrigSyncFactory.h"
#include "RecoLocalMuon/DTRecHit/interface/DTTTrigBaseSync.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"

#include "Geometry/Vector/interface/GlobalPoint.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"


#include "TFile.h"
#include "TH1F.h"

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

  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
  theFitter = new DTTimeBoxFitter();
  if(debug)
    theFitter->setVerbosity(1);

  doSubtractT0 = pset.getUntrackedParameter<bool>("doSubtractT0","false");
  // Get the synchronizer
  if(doSubtractT0) {
    theSync = DTTTrigSyncFactory::get()->create(pset.getUntrackedParameter<string>("tTrigMode"),
						pset.getUntrackedParameter<ParameterSet>("tTrigModeConfig"));
  } else {
    theSync = 0;
  }

  theTag = pset.getUntrackedParameter<string>("tTrigTag", "ttrig_test");
  checkNoisyChannels = pset.getUntrackedParameter<bool>("checkNoisyChannels","false");

  if(debug) 
    cout << "[DTTTrigCalibration]Constructor called!" << endl;
}



// DEstructor
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
  // Get the digis from the event
  Handle<DTDigiCollection> digis; 
  event.getByLabel(digiLabel, digis);



  if(doSubtractT0)
    theSync->setES(eventSetup);


  // Iterate through all digi collections ordered by LayerId   
  DTDigiCollection::DigiRangeIterator dtLayerIt;
  for (dtLayerIt = digis->begin();
       dtLayerIt != digis->end();
       ++dtLayerIt){
    // The layerId
    const DTLayerId layerId = (*dtLayerIt).first;
    const DTSuperLayerId slId = layerId.superlayerId();

    // Get the histo from the map
    TH1F *hTBox = theHistoMap[slId];
    if(hTBox == 0) {
      // Book the histogram
      theFile->cd();
      hTBox = new TH1F(getTBoxName(slId).c_str(), "Time box (ns)", 12800, -1000, 9000);
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



    // Get the iterators over the digis associated with this LayerId
    const DTDigiCollection::Range& digiRange = (*dtLayerIt).second;

    // Loop over all digis in the given range
    for (DTDigiCollection::const_iterator digi = digiRange.first;
	 digi != digiRange.second;
	 digi++) {
      const DTWireId wireId(layerId, (*digi).wire());

      // Check for noisy channels and skip them
      if(checkNoisyChannels) {
	bool isNoisy = false;
	if(setOfNoisy.find(wireId) != setOfNoisy.end())
	  isNoisy = true;

// 	bool isNoisy = false;
// 	bool isFEMasked = false;
// 	bool isTDCMasked = false;
// 	bool isTrigMask = false;
// 	bool isDead = false;
// 	bool isNohv = false;
// 	theStatusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
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

  
  // Create the object to be written to DB
  DTTtrig* tTrig = new DTTtrig(theTag);

  // Loop over the map, fit the histos and write the resulting values to the DB
  for(map<DTSuperLayerId, TH1F*>::const_iterator slHisto = theHistoMap.begin();
      slHisto != theHistoMap.end();
      slHisto++) {
    pair<double, double> meanAndSigma = theFitter->fitTimeBox((*slHisto).second);
    tTrig->setSLTtrig((*slHisto).first,
		      meanAndSigma.first,
		      meanAndSigma.second,
		      DTTimeUnits::ns);
    
    if(debug) {
      cout << " SL: " << (*slHisto).first
	   << " mean = " << meanAndSigma.first
	   << " sigma = " << meanAndSigma.second << endl;
    }
  }

  // Print the ttrig map
  dumpTTrigMap(tTrig);

  if(debug) 
   cout << "[DTTTrigCalibration]Writing ttrig object to DB!" << endl;

  // Write the ttrig object to DB
  edm::Service<cond::service::PoolDBOutputService> dbOutputSvc;
 if( dbOutputSvc.isAvailable() ){
   size_t callbackToken = dbOutputSvc->callbackToken("DTDBObject");
    try{
      dbOutputSvc->newValidityForNewPayload<DTTtrig>(tTrig, dbOutputSvc->endOfTime(), callbackToken);
    }catch(const cond::Exception& er){
      cout << er.what() << endl;
    }catch(const std::exception& er){
      cout << "[DTTTrigCalibration] caught std::exception " << er.what() << endl;
    }catch(...){
      cout << "[DTTTrigCalibration] Funny error" << endl;
    }
  }else{
    cout << "Service PoolDBOutputService is unavailable" << endl;
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
    cout << "Wh: " << (*ttrig).wheelId
	 << " St: " << (*ttrig).stationId
	 << " Sc: " << (*ttrig).sectorId
	 << " Sl: " << (*ttrig).slId
	 << " TTrig mean (ns): " << (*ttrig).tTrig * convToNs
	 << " TTrig sigma (ns): " << (*ttrig).tTrms * convToNs<< endl;
  }
}



void DTTTrigCalibration::beginJob(const edm::EventSetup& eventSetup) {
  // FIXME: this is a temporary workaraound!This should be moved to the analyze method
  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    // Get the map of noisy channels
    eventSetup.get<DTStatusFlagRcd>().get(statusMap);
  
    theStatusMap = &*statusMap;
    for(DTStatusFlag::const_iterator statusFlag = theStatusMap->begin();
	statusFlag != theStatusMap->end(); statusFlag++) {
      if((*statusFlag).noiseFlag == true) {
	
	DTWireId wireId((*statusFlag).wheelId,
			(*statusFlag).stationId,
			(*statusFlag).sectorId,
			(*statusFlag).slId,
			(*statusFlag).layerId,
			(*statusFlag).cellId);
	setOfNoisy.insert(wireId);
      }
    }

  }
}

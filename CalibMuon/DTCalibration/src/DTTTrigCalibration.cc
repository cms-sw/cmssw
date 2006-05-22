/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/05/15 10:08:42 $
 *  $Revision: 1.6 $
 *  \author G. Cerminara - INFN Torino
 */
#include "CalibMuon/DTCalibration/interface/DTDBWriterInterface.h"
#include "CalibMuon/DTCalibration/src/DTTTrigCalibration.h"
#include "CalibMuon/DTCalibration/interface/DTTimeBoxFitter.h"
#include "RecoLocalMuon/DTRecHit/interface/DTTTrigSyncFactory.h"
#include "RecoLocalMuon/DTRecHit/interface/DTTTrigBaseSync.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"

#include "Geometry/Vector/interface/GlobalPoint.h"




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
  digiLabel = pset.getParameter<string>("digiLabel");

  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
  theFitter = new DTTimeBoxFitter();
  if(debug)
    theFitter->setVerbosity(1);

  doSubtractT0 = pset.getParameter<bool>("doSubtractT0");
  // Get the synchronizer
  if(doSubtractT0) {
    theSync = DTTTrigSyncFactory::get()->create(pset.getParameter<string>("tTrigMode"),
						pset.getParameter<ParameterSet>("tTrigModeConfig"));
  } else {
    theSync = 0;
  }

  theTag = pset.getParameter<string>("tTrigTag");
  theDBWriter = new DTDBWriterInterface(pset.getParameter<ParameterSet>("dtDBWriterConfig"));


  if(debug) 
    cout << "[DTTTrigCalibration]Constructor called!" << endl;
}



// DEstructor
DTTTrigCalibration::~DTTTrigCalibration(){
  if(debug) 
    cout << "[DTTTrigCalibration]Destructor called!" << endl;

  // Delete all histos
  for(map<DTSuperLayerId, TH1F*>::const_iterator slHisto = theHistoMap.begin();
      slHisto != theHistoMap.end();
      slHisto++) {
    delete (*slHisto).second;
  }

  theFile->Close();
  delete theFitter;
  delete theDBWriter;
}



/// Perform the real analysis
void DTTTrigCalibration::analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {
  // Get the digis from the event
  Handle<DTDigiCollection> digis; 
  event.getByLabel(digiLabel, digis);

  if(doSubtractT0)
    theSync->setES(eventSetup);

  // Set the IOV of the objects FIXME: Where to put this?
  theDBWriter->setIOV(edm::IOVSyncValue::endOfTime().eventID().run());

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

    // Get the iterators over the digis associated with this LayerId
    const DTDigiCollection::Range& digiRange = (*dtLayerIt).second;

    // Loop over all digis in the given range
    for (DTDigiCollection::const_iterator digi = digiRange.first;
	 digi != digiRange.second;
	 digi++) {
      theFile->cd();
      double offset = 0;
      if(doSubtractT0) {
	const DTLayer* layer = 0;//fake
	const DTWireId wireId(layerId, (*digi).wire());
	const GlobalPoint glPt;//fake
	offset = theSync->offset(layer, wireId, glPt);
      }
      hTBox->Fill((*digi).time()-offset);
      if(debug) {
 	cout << "   Filling Time Box:   " << hTBox->GetName() << endl;
	cout << "           offset (ns): " << offset << endl;
 	cout << "           time(ns):   " << (*digi).time()-offset<< endl;
      }
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
  theDBWriter->write2DB<DTTtrig>(tTrig);



}




string DTTTrigCalibration::getTBoxName(const DTSuperLayerId& slId) const {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" << slId.wheel() << "_" << slId.station() << "_" << slId.sector()
	    << "_SL" << slId.superlayer() << "_hTimeBox";
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

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/02/21 11:15:50 $
 *  $Revision: 1.6 $
 *  \author G. Cerminara - INFN Torino
 */
#include "CalibMuon/DTCalibration/plugins/DTT0Calibration.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "CondFormats/DTObjects/interface/DTT0.h"





#include "TFile.h"
#include "TProfile.h"

class DTLayer;

using namespace std;
using namespace edm;
// using namespace cond;



// Constructor
DTT0Calibration::DTT0Calibration(const edm::ParameterSet& pset) {
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");

  // Get the label to retrieve digis from the event
  digiLabel = pset.getParameter<string>("digiLabel");

  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  if(debug) 
    cout << "[DTT0Calibration]Constructor called!" << endl;
}



// DEstructor
DTT0Calibration::~DTT0Calibration(){
  if(debug) 
    cout << "[DTT0Calibration]Destructor called!" << endl;

//   // Delete all histos
//   for(map<DTLayerId, TProfile*>::const_iterator lHisto = theHistoMap.begin();
//       lHisto != theHistoMap.end();
//       lHisto++) {
//     delete (*lHisto).second;
//   }

  theFile->Close();
}



/// Perform the real analysis
void DTT0Calibration::analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {
  if(debug | event.id().event() % 500==0)
    cout << "--- [DTT0Calibration] Analysing Event: #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;

  // Get the digis from the event
  Handle<DTDigiCollection> digis; 
  event.getByLabel(digiLabel, digis);

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);


  // Iterate through all digi collections ordered by LayerId   
  DTDigiCollection::DigiRangeIterator dtLayerIt;
  for (dtLayerIt = digis->begin();
       dtLayerIt != digis->end();
       ++dtLayerIt){
    // The layerId
    const DTLayerId layerId = (*dtLayerIt).first;

    // Get the histo from the map
    TProfile *hT0Histo = theHistoMap[layerId];
    if(hT0Histo == 0) {
      // Get the number of wires
      const DTTopology& dtTopo = dtGeom->layer(layerId)->specificTopology();
      const int nWires = dtTopo.channels();
      const int firstWire = dtTopo.firstChannel();
      const int lastWire = dtTopo.lastChannel();

      // Book the histogram
      theFile->cd();
      hT0Histo = new TProfile(getHistoName(layerId).c_str(),
			      "T0 from pulses by Channel (TDC counts, 1 TDC count = 0.781 ns)",
			      nWires, firstWire, lastWire+1);
      if(debug)
	cout << "  New T0 Histo: " << hT0Histo->GetName() << endl;
      theHistoMap[layerId] = hT0Histo;
    }

    // Get the iterators over the digis associated with this LayerId
    const DTDigiCollection::Range& digiRange = (*dtLayerIt).second;

    // Loop over all digis in the given range
    for(DTDigiCollection::const_iterator digi = digiRange.first;
	digi != digiRange.second;
	digi++) {
      theFile->cd();
      hT0Histo->Fill((*digi).wire(), (*digi).countsTDC());
      //hT0Histo->Fill((*digi).wire(), (*digi).wire());
      if(debug) {
 	cout << "   Wire: " << DTWireId(layerId, (*digi).wire()) << endl
	     << "       time (TDC counts): " << (*digi).countsTDC()<< endl;
      }
    }
  }

}


void DTT0Calibration::endJob() {
  if(debug) 
    cout << "[DTT0Calibration]Writing histos to file!" << endl;
  
  // Write all histos to file
  theFile->cd();
  for(map<DTLayerId, TProfile*>::const_iterator lHisto = theHistoMap.begin();
      lHisto != theHistoMap.end();
      lHisto++) {
    (*lHisto).second->Write();
  }
  
  // Create the object to be written to DB
  DTT0* t0s = new DTT0(); //FIXME: do we need a tag????

  // Normalization to chamber mean
  map<DTChamberId, pair<double, int> > chamberMeanMap;
  // Loop over the map, get mean from the histos 
   for(map<DTLayerId, TProfile*>::const_iterator lHisto = theHistoMap.begin();
      lHisto != theHistoMap.end();
      lHisto++) {
     // FIXME: initialization?
     pair<double, int> meanAndN = chamberMeanMap[(*lHisto).first.chamberId()];
     TProfile *hist = (*lHisto).second;
     int nBins = hist->GetNbinsX();
     // Loop over all the channels
     for(int bin = 1; bin <= nBins; bin++) {
       float t0mean = hist->GetBinContent(bin);
       if(t0mean != 0) {
	 meanAndN.first = (meanAndN.first * meanAndN.second + t0mean)/(++meanAndN.second);
// 	 if(debug) {
// 	   cout << " Chamber: " << (*lHisto).first.chamberId() << endl
// 		<< "     new Mean (TDC counts): " << meanAndN.first << endl
// 		<< "     N entries: " << meanAndN.second << endl;
// 	 }
       }
     }
     chamberMeanMap[(*lHisto).first.chamberId()] = meanAndN;
   }
   
   

  // Loop over the map, get mean and RMS from the histos and write the resulting values to the DB
  for(map<DTLayerId, TProfile*>::const_iterator lHisto = theHistoMap.begin();
      lHisto != theHistoMap.end();
      lHisto++) {
    TProfile *hist = (*lHisto).second;
    int nBins = hist->GetNbinsX();
    int channel =  (int)(hist->GetXaxis()->GetXmin());
    // Loop over all the channels
    for(int bin = 1; bin <= nBins; bin++) {
      float t0mean = hist->GetBinContent(bin);
      float t0rms = hist->GetBinError(bin);
      DTWireId wireId((*lHisto).first, channel);
      // Write the channel t0 normalized to the chamber mean value
      t0s->setCellT0(wireId, t0mean-chamberMeanMap[wireId.chamberId()].first, t0rms);
      channel++;

      if(debug) {
	cout << " Wire: " << wireId
	     << " mean = " << t0mean
	     << " rms = " << t0rms << endl;
      }
    }
  }

  // Print the t0 map
  dumpT0Map(t0s);

  if(debug) 
   cout << "[DTT0Calibration]Writing ttrig object to DB!" << endl;

  // FIXME: to be read from cfg?
  string t0Record = "DTT0Rcd";

  // Write the t0 map to DB
  DTCalibDBUtils::writeToDB(t0Record, t0s);
}



   

string DTT0Calibration::getHistoName(const DTLayerId& lId) const {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" << lId.wheel() << "_" << lId.station() << "_" << lId.sector()
	    << "_SL" << lId.superlayer() << "_L" << lId.layer() << "_hT0Histo";
  theStream >> histoName;
  return histoName;
}


void DTT0Calibration::dumpT0Map(const DTT0* t0s) const {
//   static const double convToNs = 25./32.;
  for(DTT0::const_iterator t0 = t0s->begin();
      t0 != t0s->end(); t0++) {
    cout << "Wh: " << (*t0).first.wheelId
	 << " St: " << (*t0).first.stationId
	 << " Sc: " << (*t0).first.sectorId
	 << " Sl: " << (*t0).first.slId
	 << " L:  " << (*t0).first.layerId
	 << " Wi: " << (*t0).first.cellId
	 << " T0 mean (TDC counts): " << (*t0).second.t0mean
	 << " T0 sigma (TDC counts): " << (*t0).second.t0rms << endl;
  }
}

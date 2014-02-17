
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/05/11 17:17:17 $
 *  $Revision: 1.9 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "DTT0Analyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include <iostream>
#include "TFile.h"
#include "TH1D.h"
#include "TString.h"

using namespace edm;
using namespace std;

DTT0Analyzer::DTT0Analyzer(const ParameterSet& pset) {
  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
 
}
 
DTT0Analyzer::~DTT0Analyzer(){  
  theFile->Close();
}

void DTT0Analyzer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup) {
  //Get the t0 map from the DB
  ESHandle<DTT0> t0;
  eventSetup.get<DTT0Rcd>().get(t0);
  tZeroMap = &*t0;

  // Get the DT Geometry  
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);
}

void DTT0Analyzer::endJob() {
  // Loop over DB entries
  for(DTT0::const_iterator tzero = tZeroMap->begin();
      tzero != tZeroMap->end(); tzero++) {
// @@@ NEW DTT0 FORMAT
//    DTWireId wireId((*tzero).first.wheelId,
//		    (*tzero).first.stationId,
//		    (*tzero).first.sectorId,
//		    (*tzero).first.slId,
//		    (*tzero).first.layerId,
//		    (*tzero).first.cellId);
    int channelId = tzero->channelId;
    if ( channelId == 0 ) continue;
    DTWireId wireId( channelId );
// @@@ NEW DTT0 END
    float t0mean;
    float t0rms;
    // t0s and rms are TDC counts
    tZeroMap->get(wireId, t0mean, t0rms, DTTimeUnits::counts);
    cout << "Wire: " <<  wireId <<endl
	 << " T0 mean (TDC counts): " << t0mean
	 << " T0_rms (TDC counts): " << t0rms << endl;

    DTLayerId layerId = wireId.layerId();
    const int nWires = dtGeom->layer(layerId)->specificTopology().channels();


    //Define an histo for means and an histo for sigmas for each layer
    TH1D *hT0Histo = theMeanHistoMap[layerId];
    if(hT0Histo == 0) {
      theFile->cd();
      TString name = getHistoName(layerId).c_str();
      hT0Histo = new TH1D(name+"_t0Mean",
			  "mean T0 from pulses by Channel", nWires,dtGeom->layer(layerId)->specificTopology().firstChannel(),
			  dtGeom->layer(layerId)->specificTopology().firstChannel()+nWires);
      theMeanHistoMap[layerId] = hT0Histo;
     }

    TH1D *hSigmaT0Histo = theSigmaHistoMap[layerId];
    if(hSigmaT0Histo == 0) {
      theFile->cd();
      TString name = getHistoName(layerId).c_str();
      hSigmaT0Histo = new TH1D(name+"_t0Sigma",
			  "sigma T0 from pulses by Channel", nWires,dtGeom->layer(layerId)->specificTopology().firstChannel(),
			  dtGeom->layer(layerId)->specificTopology().firstChannel()+nWires);
      theSigmaHistoMap[layerId] = hSigmaT0Histo;
     }

    //Fill the histos
    int nBin= hT0Histo->GetXaxis()->FindFixBin(wireId.wire());
    hT0Histo->SetBinContent(nBin,t0mean);  
    hSigmaT0Histo->SetBinContent(nBin,t0rms);  
  }

  //Write histos in a .root file
  theFile->cd();
  for(map<DTLayerId, TH1D*>::const_iterator lHisto = theMeanHistoMap.begin();
      lHisto != theMeanHistoMap.end();
      lHisto++) {
    (*lHisto).second->Write(); 
  }    

  for(map<DTLayerId, TH1D*>::const_iterator lHisto = theSigmaHistoMap.begin();
      lHisto != theSigmaHistoMap.end();
      lHisto++) {
    (*lHisto).second->Write(); 
 }

}

string DTT0Analyzer::getHistoName(const DTLayerId& lId) const {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" << lId.wheel() << "_" << lId.station() << "_" << lId.sector()
	    << "_SL" << lId.superlayer() << "_L" << lId.layer();
  theStream >> histoName;
  return histoName;
}

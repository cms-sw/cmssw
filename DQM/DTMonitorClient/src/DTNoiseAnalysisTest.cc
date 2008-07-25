

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/07/22 10:23:26 $
 *  $Revision: 1.6 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTNoiseAnalysisTest.h>

// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>



using namespace edm;
using namespace std;


DTNoiseAnalysisTest::DTNoiseAnalysisTest(const edm::ParameterSet& ps){

  edm::LogVerbatim ("noise") << "[DTNoiseAnalysisTest]: Constructor";

  dbe = edm::Service<DQMStore>().operator->();

  // get the cfi parameters
  noisyCellDef = ps.getUntrackedParameter<int>("noisyCellDef",500);

}


DTNoiseAnalysisTest::~DTNoiseAnalysisTest(){

  edm::LogVerbatim ("noise") << "DTNoiseAnalysisTest: analyzed " << nevents << " events";
}


void DTNoiseAnalysisTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("noise") <<"[DTNoiseAnalysisTest]: BeginJob"; 

  nevents = 0;
  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // book the histos
  bookHistos();  

}


void DTNoiseAnalysisTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("noise") <<"[DTNoiseAnalysisTest]: Begin of LS transition";
}


void DTNoiseAnalysisTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("noise") << "[DTNoiseAnalysisTest]: "<<nevents<<" events";

}

void DTNoiseAnalysisTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  edm::LogVerbatim ("noise") <<"[DTNoiseAnalysisTest]: End of LS transition, performing the DQM client operation";

  // Reset the summary plots
  for(map<int, MonitorElement* >::iterator plot =  noiseHistos.begin();
      plot != noiseHistos.end(); ++plot) {
    (*plot).second->Reset();
  }
  
  for(map<int,  MonitorElement* >::iterator plot = noisyCellHistos.begin();
      plot != noisyCellHistos.end(); ++plot) {
    (*plot).second->Reset();
  }

  summaryNoiseHisto->Reset();



  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();
  
  edm::LogVerbatim ("noise") <<"[DTNoiseAnalysisTest]: Fill the summary histos";

  for (; ch_it != ch_end; ++ch_it) { // loop over chambers
    DTChamberId chID = (*ch_it)->id();
    
    MonitorElement * histo = dbe->get(getMEName(chID));
    
    if(histo){ // check the pointer
      
      TH2F * histo_root = histo->getTH2F();
      
      for(int sl = 1; sl != 4; ++sl) { // loop over SLs
	// skip theta SL in MB4 chambers
	if(chID.station() == 4 && sl == 2) continue;

	int binYlow = ((sl-1)*4)+1;

	for(int layer = 1; layer <= 4; ++layer) { // loop over layers
	  
	  // Get the layer ID
	  DTLayerId layID(chID,sl,layer);

	  int nWires = muonGeom->layer(layID)->specificTopology().channels();
	  int firstWire = muonGeom->layer(layID)->specificTopology().firstChannel();

	  int binY = binYlow+(layer-1);

	  for(int wire = firstWire; wire != (nWires+firstWire); wire++){ // loop over wires

	    double noise = histo_root->GetBinContent(wire, binY);
	    // fill the histos
	    noiseHistos[chID.wheel()]->Fill(noise);
	    noiseHistos[3]->Fill(noise);
	    int sector = chID.sector();
	    if(noise>noisyCellDef) {
	      if(sector == 13) {
		sector = 4;
	      } else if(sector == 14) {
		sector = 10;
	      }
	      noisyCellHistos[chID.wheel()]->Fill(sector,chID.station());
	      summaryNoiseHisto->Fill(sector,chID.wheel());
	    }
	  }
	}
      }
    }
  }
}	       


string DTNoiseAnalysisTest::getMEName(const DTChamberId & chID) {
  
  stringstream wheel; wheel << chID.wheel();	
  stringstream station; station << chID.station();	
  stringstream sector; sector << chID.sector();	
  
  string folderName = 
    "DT/04-Noise/Wheel" +  wheel.str() +
//     "/Station" + station.str() +
    "/Sector" + sector.str() + "/";

  string histoname = folderName + string("NoiseRate")  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str();
  
  return histoname;
  
}


void DTNoiseAnalysisTest::bookHistos() {
  
  dbe->setCurrentFolder("DT/04-Noise");
  string histoName;

  for(int wh=-2; wh<=2; wh++){
      stringstream wheel; wheel << wh;
      histoName =  "NoiseRateSummary_W" + wheel.str();
      noiseHistos[wh] = dbe->book1D(histoName.c_str(),histoName.c_str(),100,0,2000);
      noiseHistos[wh]->setAxisTitle("rate (Hz)",1);
      noiseHistos[wh]->setAxisTitle("entries",2);
  }
  histoName =  "NoiseRateSummary";
  noiseHistos[3] = dbe->book1D(histoName.c_str(),histoName.c_str(),100,0,2000);
  noiseHistos[3]->setAxisTitle("rate (Hz)",1);
  noiseHistos[3]->setAxisTitle("entries",2);

  
  for(int wh=-2; wh<=2; wh++){
    stringstream wheel; wheel << wh;
    histoName =  "NoiseSummary_W" + wheel.str();
    noisyCellHistos[wh] = dbe->book2D(histoName.c_str(),"# of noisy channels",12,1,13,4,1,5);
    noisyCellHistos[wh]->setBinLabel(1,"MB1",2);
    noisyCellHistos[wh]->setBinLabel(2,"MB2",2);
    noisyCellHistos[wh]->setBinLabel(3,"MB3",2);
    noisyCellHistos[wh]->setBinLabel(4,"MB4",2);  
    noisyCellHistos[wh]->setAxisTitle("Sector",1);
  }

  histoName =  "NoiseSummary";
  summaryNoiseHisto =  dbe->book2D(histoName.c_str(),"# of noisy channels",12,1,13,5,-2,3);
  summaryNoiseHisto->setAxisTitle("Sector",1);
  summaryNoiseHisto->setAxisTitle("Wheel",2);

}


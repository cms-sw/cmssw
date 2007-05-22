/*
 * 
 * $Date: 2007/05/15 17:21:35 $
 * $Revision: 1.3 $
 * \author A. Gresele - INFN Trento
 *
 */

#include "DQM/DTMonitorClient/src/DTNoiseTest.h"

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DQMServices/Core/interface/MonitorElementBaseT.h>

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
#include <vector>

using namespace edm;
using namespace std;

DTNoiseTest::DTNoiseTest(const edm::ParameterSet& ps){
  
  edm::LogVerbatim ("noise") <<"[DTNoiseTest]: Constructor";

  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);
  dbe->setCurrentFolder("DT/TestNoise");
}


DTNoiseTest::~DTNoiseTest(){

  edm::LogVerbatim ("noise") <<"DTNoiseTest: analyzed " << updates << " events";

}

void DTNoiseTest::endJob(){

  edm::LogVerbatim ("noise") <<"[DTNoiseTest] endjob called!";

  if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ) 
    dbe->save(parameters.getUntrackedParameter<string>("outputFile", "DTNoiseTest.root"));
  
  dbe->rmdir("DT/TestNoise");
}

void DTNoiseTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("noise") <<"[DTNoiseTest]: BeginJob";

  updates = 0;
  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTNoiseTest::bookHistos(const DTChamberId & ch, string folder, string histoTag ) {

  stringstream wheel; wheel << ch.wheel();	
  stringstream station; station << ch.station();	
  stringstream sector; sector << ch.sector();	

  dbe->setCurrentFolder("DT/TestNoise/" + folder);

  string histoName =  histoTag + "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str(); 
 
  if ( folder == "Dead Channels")
  (histos[histoTag])[ch.rawId()] = dbe->book1D(histoName.c_str(),histoName.c_str(),3,0,3);
 
  if ( folder == "New Noisy Channels")
  (histos[histoTag])[ch.rawId()] = dbe->book1D(histoName.c_str(),histoName.c_str(),3,0,3);
  
}


void DTNoiseTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  updates++;

  edm::LogVerbatim ("noise") <<"[DTNoiseTest]: "<<updates<<" updates";

  ESHandle<DTStatusFlag> statusMap;
  context.get<DTStatusFlagRcd>().get(statusMap);
  
  context.get<DTTtrigRcd>().get(tTrigMap);
  float tTrig, tTrigRMS;

  string histoTag;
  
  // loop over chambers
  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  for (; ch_it != ch_end; ++ch_it) {
    
    DTChamberId ch = (*ch_it)->id();
  
    MonitorElement * noiseME = dbe->get(getMEName(ch));
    
    if (noiseME) {
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>(noiseME);
      if (ob) {
	TH2F * noiseHisto = dynamic_cast<TH2F*> (ob->operator->());
	
	double nevents=noiseHisto->GetEntries();
	
	double normalization =0;
	
	if (noiseHisto) {
	  
	  // loop over SLs
	  vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 
	  vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();
	  
	  float average=0;
	  float nOfChannels=0;
	  float noiseStatistics=0;
	  int newNoiseChannels=0;

	  for(; sl_it != sl_end; ++sl_it) {
	    
	    const DTSuperLayerId & slID = (*sl_it)->id();
	    
	    tTrigMap->slTtrig(slID, tTrig, tTrigRMS);
	    if (tTrig==0) tTrig=1;
	    const double ns_s = 1e9*(32/25);
	    normalization = ns_s/float(tTrig*nevents);
	    
	    noiseHisto->Scale(normalization);
	    
	    // loop over layers
	    
	    for (int binY=(slID.superLayer()-1)*4+1 ; binY <= (slID.superLayer()-1)*4+4; binY++) {
	      
	      int Y = binY - 4*(slID.superLayer()-1);
	      
	      // the layer
	      
	      const DTLayerId theLayer(slID,Y);
	      
	      // loop over channels 
	      for (int binX=1; binX <= noiseHisto->GetNbinsX(); binX++) {
		
		if (noiseHisto->GetBinContent(binX,binY) > parameters.getUntrackedParameter<int>("HzThreshold", 300))
		  theNoisyChannels.push_back(DTWireId(theLayer, binX));
		  
		// get rid of the dead channels
		else {
		  average += noiseHisto->GetBinContent(binX,binY); 
		  nOfChannels++; 
		}
	      }
	    }
	    
	    if (nOfChannels) noiseStatistics = average/nOfChannels;
	    	    
	    histoTag = "Dead Channels";

	    if (histos[histoTag].find((*ch_it)->id().rawId()) == histos[histoTag].end()) bookHistos((*ch_it)->id(),string("Dead Channels"), histoTag );
	    histos[histoTag].find((*ch_it)->id().rawId())->second->setBinContent(slID.superLayer(),noiseStatistics); 
	    
	    for ( vector<DTWireId>::const_iterator nb_it = theNoisyChannels.begin();
		  nb_it != theNoisyChannels.end(); ++nb_it) {
	      
	      bool isNoisy = false;
	      bool isFEMasked = false;
	      bool isTDCMasked = false;
	      bool isTrigMask = false;
	      bool isDead = false;
	      bool isNohv = false;
	      statusMap->cellStatus((*nb_it), isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
	    	      
	      if (!isNoisy) newNoiseChannels++;
	    }

	    theNoisyChannels.clear();

	    histoTag = "New Noisy Channels";
	    if (histos[histoTag].find((*ch_it)->id().rawId()) == histos[histoTag].end()) bookHistos((*ch_it)->id(),string("New Noisy Channels"), histoTag );
	    histos[histoTag].find((*ch_it)->id().rawId())->second->setBinContent(slID.superLayer(), newNoiseChannels); 
	    
	  }
	}
      }
    } 
  }
}

string DTNoiseTest::getMEName(const DTChamberId & ch) {
  
  stringstream wheel; wheel << ch.wheel();	
  stringstream station; station << ch.station();	
  stringstream sector; sector << ch.sector();	
  
  string folderTag = parameters.getUntrackedParameter<string>("folderTag", "Occupancies");
  string folderName = 
    "Collector/FU0/DT/DTDigiTask/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/" + folderTag + "/";
  
  string histoTag = parameters.getUntrackedParameter<string>("histoTag", "OccupancyInTimeHitsNoise");
  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 
    
    
  return histoname;
  
}

/*
 * 
 * \authors:
 *  A. Gresele - INFN Trento
 *  G. Mila - INFN Torino
 *  M. Zanetti - CERN PH
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 *
 */

#include "DQM/DTMonitorClient/src/DTNoiseTest.h"

// Framework
#include <FWCore/Framework/interface/EventSetup.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;



DTNoiseTest::DTNoiseTest(const edm::ParameterSet& ps){

  edm::LogVerbatim ("noise") <<"[DTNoiseTest]: Constructor";  

  parameters = ps;
  
  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

  bookingdone = 0;

}



DTNoiseTest::~DTNoiseTest(){

  edm::LogVerbatim ("noise") <<"DTNoiseTest: analyzed " << updates << " events";

}

  void DTNoiseTest::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
                             edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & context) {

  if (!bookingdone) {
    ibooker.setCurrentFolder("DT/Tests/Noise");
  updates = 0;
  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);
  }
  bookingdone = 1; 

  edm::LogVerbatim ("noise") <<"[DTNoiseTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  edm::LogVerbatim ("noise") <<"[DTNoiseTest]: "<<nLumiSegs<<" updates";

  ESHandle<DTStatusFlag> statusMap;
  context.get<DTStatusFlagRcd>().get(statusMap);
  
  context.get<DTTtrigRcd>().get(tTrigMap);
  float tTrig, tTrigRMS, kFactor;

  string histoTag;
  // loop over chambers
  vector<const DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId ch = (*ch_it)->id();
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();
	    
    MonitorElement * noiseME = igetter.get(getMEName(ch));
    if (noiseME) {
      TH2F * noiseHisto = noiseME->getTH2F();

      // WARNING uncorrect normalization!! TO BE PROVIDED CENTRALLY
      double nevents = (int) noiseHisto->GetEntries();	
	
      double normalization =0;

      float average=0;
      float nOfChannels=0;
      float noiseStatistics=0;
      int newNoiseChannels=0;

      for(; sl_it != sl_end; ++sl_it) {
	const DTSuperLayerId & slID = (*sl_it)->id();
	    
        // ttrig and rms are counts
	tTrigMap->get(slID, tTrig, tTrigRMS, kFactor, DTTimeUnits::counts);
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
	histoTag = "NoiseAverage";

	if (histos[histoTag].find((*ch_it)->id().rawId()) == histos[histoTag].end()) bookHistos(ibooker,(*ch_it)->id(),string("NoiseAverage"), histoTag );
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
	histoTag = "NewNoisyChannels";

	if (histos[histoTag].find((*ch_it)->id().rawId()) == histos[histoTag].end()) 
               bookHistos(ibooker,(*ch_it)->id(),string("NewNoisyChannels"), histoTag );
	histos[histoTag].find((*ch_it)->id().rawId())->second->setBinContent(slID.superLayer(), newNoiseChannels);   
      }
    }
    //To compute the Noise Mean test
    vector<const DTSuperLayer*>::const_iterator sl2_it = (*ch_it)->superLayers().begin(); 
    vector<const DTSuperLayer*>::const_iterator sl2_end = (*ch_it)->superLayers().end();
    for(; sl2_it != sl2_end; ++sl2_it) {
      vector<const DTLayer*>::const_iterator l_it = (*sl2_it)->layers().begin(); 
      vector<const DTLayer*>::const_iterator l_end = (*sl2_it)->layers().end();
      for(; l_it != l_end; ++l_it) {
	
	DTLayerId lID = (*l_it)->id();
	MonitorElement * noisePerEventME = igetter.get(getMEName(lID));

	if (noisePerEventME) {
	  TH2F * noiseHistoPerEvent = noisePerEventME->getTH2F();
	  int nWires = muonGeom->layer(lID)->specificTopology().channels();
	  double MeanNumerator=0, MeanDenominator=0;
	  histoTag = "MeanDigiPerEvent";
	  for (int w=1; w<=nWires; w++){
	    for(int numDigi=1; numDigi<=10; numDigi++){
	      MeanNumerator+=(noiseHistoPerEvent->GetBinContent(w,numDigi)*(numDigi-1));
	      MeanDenominator+=noiseHistoPerEvent->GetBinContent(w,numDigi);
	    }
	    double Mean=MeanNumerator/MeanDenominator;
	    if (histos[histoTag].find((*l_it)->id().rawId()) == histos[histoTag].end()) bookHistos(ibooker,(*l_it)->id(),nWires, string("MeanDigiPerEvent"), histoTag );
	    histos[histoTag].find((*l_it)->id().rawId())->second->setBinContent(w, Mean);   
	  } 
	}
      }
    }
  }
  
  // Noise Mean test 
  histoTag = "MeanDigiPerEvent";
  string MeanCriterionName = parameters.getUntrackedParameter<string>("meanTestName","NoiseMeanInRange");
  for(map<uint32_t, MonitorElement*>::const_iterator hMean = histos[histoTag].begin();
      hMean != histos[histoTag].end();
      hMean++) {
    const QReport * theMeanQReport = (*hMean).second->getQReport(MeanCriterionName);
    if(theMeanQReport) {
      vector<dqm::me_util::Channel> badChannels = theMeanQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	LogVerbatim ("tTrigCalibration")<<"LayerId : "<<(*hMean).first<<" Bad mean channels: "
                                        <<(*channel).getBin()<<"  Contents : "<<(*channel).getContents();
      }
    }
  }
  
}

void DTNoiseTest::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  edm::LogVerbatim ("noise") <<"[DTNoiseTest] endjob called!";
}


string DTNoiseTest::getMEName(const DTChamberId & ch) {
  
  stringstream wheel; wheel << ch.wheel();	
  stringstream station; station << ch.station();	
  stringstream sector; sector << ch.sector();	
  
  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderTag = parameters.getUntrackedParameter<string>("folderTag", "Occupancies");
  string folderName = 
    folderRoot + "DT/DTDigiTask/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/" + folderTag + "/";

  string histoTag = parameters.getUntrackedParameter<string>("histoTag", "OccupancyNoise_perCh");
  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 
    
    
  return histoname;
  
}

string DTNoiseTest::getMEName(const DTLayerId & ly) {
  
  stringstream wheel; wheel << ly.wheel();	
  stringstream station; station << ly.station();	
  stringstream sector; sector << ly.sector();
  stringstream superLayer; superLayer << ly.superlayer();
  stringstream layer; layer << ly.layer();
  
  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderTag = parameters.getUntrackedParameter<string>("folderTagForDigiPerEventTest", "DigiPerEvent");
  string folderName = 
    folderRoot + "DT/DTDigiForNoiseTask/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/" + folderTag + "/";
  
  string histoTag = parameters.getUntrackedParameter<string>("histoTagForDigiPerEventTest", "DigiPerEvent");
  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str()
    + "_SL" + superLayer.str()
    + "_L" + layer.str();
    
    
  return histoname;

}

void DTNoiseTest::bookHistos(DQMStore::IBooker & ibooker, const DTChamberId & ch, string folder, string histoTag ) {

  stringstream wheel; wheel << ch.wheel();	
  stringstream station; station << ch.station();	
  stringstream sector; sector << ch.sector();	

  ibooker.setCurrentFolder("DT/Tests/Noise/" + folder);

  string histoName =  histoTag + "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str(); 
 
  if (folder == "NoiseAverage")
  (histos[histoTag])[ch.rawId()] = ibooker.book1D(histoName.c_str(),histoName.c_str(),3,0,3);
 
  if ( folder == "NewNoisyChannels")
  (histos[histoTag])[ch.rawId()] = ibooker.book1D(histoName.c_str(),histoName.c_str(),3,0,3);
  
}

void DTNoiseTest::bookHistos(DQMStore::IBooker & ibooker, const DTLayerId & lId, int nWires, string folder, string histoTag) {

  stringstream wheel; wheel << lId.superlayerId().wheel();
  stringstream station; station << lId.superlayerId().station();	
  stringstream sector; sector << lId.superlayerId().sector();
  stringstream superLayer; superLayer << lId.superlayerId().superlayer();
  stringstream layer; layer << lId.layer();

  string histoName = histoTag + "_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() 
                              +  "_SL" + superLayer.str() +  "_L" + layer.str();

  ibooker.setCurrentFolder("DT/Tests/Noise/" + folder +
			"/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str());

  (histos[histoTag])[lId.rawId()] = ibooker.book1D(histoName.c_str(),histoName.c_str(),nWires,0,nWires);
}

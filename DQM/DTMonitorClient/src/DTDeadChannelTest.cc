
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.15 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTDeadChannelTest.h>

// Framework
#include <FWCore/Framework/interface/EventSetup.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <stdio.h>
#include <sstream>
#include <math.h>


using namespace edm;
using namespace std;


DTDeadChannelTest::DTDeadChannelTest(const edm::ParameterSet& ps){
 
  edm::LogVerbatim ("deadChannel") << "[DTDeadChannelTest]: Constructor";

  parameters = ps;

  dbe = edm::Service<DQMStore>().operator->();

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

}

DTDeadChannelTest::~DTDeadChannelTest(){

  edm::LogVerbatim ("deadChannel") << "DTDeadChannelTest: analyzed " << nevents << " events";

}


void DTDeadChannelTest::beginJob(){

  edm::LogVerbatim ("deadChannel") << "[DTDeadChannelTest]: BeginJob";

  nevents = 0;

}

void DTDeadChannelTest::beginRun(Run const& run, EventSetup const& context) {

  edm::LogVerbatim ("deadChannel") << "[DTDeadChannelTest]: BeginRun";

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTDeadChannelTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("deadChannel") <<"[DTDeadChannelTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

}



void DTDeadChannelTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("deadChannel") << "[DTDeadChannelTest]: "<<nevents<<" events";

}



void DTDeadChannelTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  // counts number of updats (online mode) or number of events (standalone mode)
  //nevents++;
  // if running in standalone perform diagnostic only after a reasonalbe amount of events
  //if ( parameters.getUntrackedParameter<bool>("runningStandalone", false) && 
  //     nevents%parameters.getUntrackedParameter<int>("diagnosticPrescale", 1000) != 0 ) return;
  //edm::LogVerbatim ("deadChannel") << "[DTDeadChannelTest]: "<<nevents<<" updates";


  edm::LogVerbatim ("deadChannel") <<"[DTDeadChannelTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  edm::LogVerbatim ("deadChannel") <<"[DTDeadChannelTest]: "<<nLumiSegs<<" updates";


  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  edm::LogVerbatim ("deadChannel") << "[DTDeadChannelTest]: Occupancy tests results";

  // Loop over the chambers
  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId chID = (*ch_it)->id();
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();

    stringstream wheel; wheel << chID.wheel();
    stringstream station; station << chID.station();
    stringstream sector; sector << chID.sector();
    
    context.get<DTTtrigRcd>().get(tTrigMap);

    string HistoName = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str(); 

    // Get the ME produced by DigiTask Source
    MonitorElement * noise_histo = dbe->get(getMEName("OccupancyNoise_perCh", chID));	
    MonitorElement * hitInTime_histo = dbe->get(getMEName("OccupancyInTimeHits_perCh", chID));

    // ME -> TH2F
    if(noise_histo && hitInTime_histo) {	  
      TH2F * noise_histo_root = noise_histo->getTH2F();
      TH2F * hitInTime_histo_root = hitInTime_histo->getTH2F();

      // Loop over the SuperLayers
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId slID = (*sl_it)->id();
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin();
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
	    
        // ttrig and rms are counts
	float tTrig, tTrigRMS, kFactor;
	tTrigMap->get(slID, tTrig, tTrigRMS, kFactor, DTTimeUnits::counts);
      
	// Loop over the layers
	for(; l_it != l_end; ++l_it) {
	  DTLayerId lID = (*l_it)->id();

	  //Parameters to fill histos
	  stringstream superLayer; superLayer << slID.superlayer();
	  stringstream layer; layer << lID.layer();
	  string HistoNameTest = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() +  "_SL" + superLayer.str() +  "_L" + layer.str();

	  const int firstWire = muonGeom->layer(lID)->specificTopology().firstChannel();
	  const int lastWire = muonGeom->layer(lID)->specificTopology().lastChannel();

	  int entry=-1;
	  if(slID.superlayer() == 1) entry=0;
	  if(slID.superlayer() == 2) entry=4;
	  if(slID.superlayer() == 3) entry=8;
	  int YBinNumber = entry+lID.layer();
	      

	  // Loop over the TH2F bin and fill the ME to be used for the Quality Test
	  for(int bin=firstWire; bin <= lastWire; bin++) {
	    if (OccupancyDiffHistos.find(HistoNameTest) == OccupancyDiffHistos.end()) bookHistos(lID, firstWire, lastWire);
	    // tMax default value
	    float tMax = 450.0;

	    float difference = (hitInTime_histo_root->GetBinContent(bin, YBinNumber) / tMax) 
			       - (noise_histo_root->GetBinContent(bin, YBinNumber) / tTrig);
	    OccupancyDiffHistos.find(HistoNameTest)->second->setBinContent(bin, difference);
	  }
	} // loop on layers
      } // loop on superlayers
    }
  } // loop on chambers

  // Occupancy Difference test 
  string OccupancyDiffCriterionName = parameters.getUntrackedParameter<string>("OccupancyDiffTestName","OccupancyDiffInRange"); 
  for(map<string, MonitorElement*>::const_iterator hOccDiff = OccupancyDiffHistos.begin();
      hOccDiff != OccupancyDiffHistos.end();
      hOccDiff++) {
    const QReport * theOccupancyDiffQReport = (*hOccDiff).second->getQReport(OccupancyDiffCriterionName);
    if(theOccupancyDiffQReport) {
      vector<dqm::me_util::Channel> badChannels = theOccupancyDiffQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	edm::LogError ("deadChannel") << "Layer : "<<(*hOccDiff).first<<" Bad occupancy difference channels: "<<(*channel).getBin()<<" Contents : "<<(*channel).getContents();
      }
      // FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
      // edm::LogWarning("deadChannel")<< "-------- Layer : "<<(*hOccDiff).first<<"  "<<theOccupancyDiffQReport->getMessage()<<" ------- "<<theOccupancyDiffQReport->getStatus(); 
    }
  }

}


void DTDeadChannelTest::endJob(){

  edm::LogVerbatim ("deadChannel") << "[DTDeadChannelTest] endjob called!";

  dbe->rmdir("DT/Tests/DTDeadChannel");

}

string DTDeadChannelTest::getMEName(string histoTag, const DTChamberId & chId) {

  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();
  stringstream sector; sector << chId.sector();

  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderName = 
    folderRoot + "DT/DTDigiTask/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + 
    "/Occupancies" + "/";
  
  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str();
  
  return histoname;
  
}


void DTDeadChannelTest::bookHistos(const DTLayerId & lId, int firstWire, int lastWire) {

  stringstream wheel; wheel << lId.superlayerId().wheel();
  stringstream station; station << lId.superlayerId().station();	
  stringstream sector; sector << lId.superlayerId().sector();
  stringstream superLayer; superLayer << lId.superlayerId().superlayer();
  stringstream layer; layer << lId.layer();

  string HistoName = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() +  "_SL" + superLayer.str() +  "_L" + layer.str();
  string OccupancyDiffHistoName =  "OccupancyDiff_" + HistoName; 

  dbe->setCurrentFolder("DT/Tests/DTDeadChannel/Wheel" + wheel.str() +
			   "/Station" + station.str() +
			   "/Sector" + sector.str());

  OccupancyDiffHistos[HistoName] = dbe->book1D(OccupancyDiffHistoName.c_str(),OccupancyDiffHistoName.c_str(),lastWire-firstWire+1, firstWire-0.5, lastWire+0.5);

}

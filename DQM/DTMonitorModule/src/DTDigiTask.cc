 /*
 * \file DTDigiTask.cc
 * 
 * $Date: 2008/05/22 07:22:54 $
 * $Revision: 1.42 $
 * \author M. Zanetti - INFN Padova
 *
 */

#include <DQM/DTMonitorModule/interface/DTDigiTask.h>

// Framework
#include <FWCore/Framework/interface/EventSetup.h>

// Digis
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// T0s
#include <CondFormats/DTObjects/interface/DTT0.h>
#include <CondFormats/DataRecord/interface/DTT0Rcd.h>
#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;



// Contructor
DTDigiTask::DTDigiTask(const edm::ParameterSet& ps){
  // switch for the verbosity
  debug = ps.getUntrackedParameter<bool>("debug", "false");
  if(debug) cout<<"[DTDigiTask]: Constructor"<<endl;

  // The label to retrieve the digis 
  dtDigiLabel = ps.getParameter<InputTag>("dtDigiLabel");
  // Read the configuration parameters
  maxTDCHits = ps.getUntrackedParameter<int>("maxTDCHitsPerChamber",30000);
  // Set to true to read the ttrig from DB (useful to determine in-time and out-of-time hits)
  readTTrigDB = ps.getUntrackedParameter<bool>("readDB", false);
  // Set to true to subtract t0 from test pulses
  subtractT0 = ps.getParameter<bool>("performPerWireT0Calibration");
  // Tmax value (TDC counts)
  defaultTmax = ps.getParameter<int>("defaultTmax");
  // Switch from static to dinamic histo booking
  doStaticBooking =  ps.getUntrackedParameter<bool>("staticBooking", true);
  // Switch for local/global runs
  isLocalRun = ps.getUntrackedParameter<bool>("localrun", true);
  // Setting for the reset of the ME after n (= ResetCycle) luminosity sections
  resetCycle = ps.getUntrackedParameter<int>("ResetCycle", 3);
  // Check the DB of noisy channels
  checkNoisyChannels = ps.getUntrackedParameter<bool>("checkNoisyChannels","false");
  // Default TTrig to be used when not reading the TTrig DB
  defaultTTrig = ps.getParameter<int>("defaultTtrig");
  inTimeHitsLowerBound = ps.getParameter<int>("inTimeHitsLowerBound");
  inTimeHitsUpperBound = ps.getParameter<int>("inTimeHitsUpperBound");
  timeBoxGranularity = ps.getUntrackedParameter<int>("timeBoxGranularity",4);
  tdcRescale = ps.getUntrackedParameter<int>("tdcRescale", 1);

  
  dbe = edm::Service<DQMStore>().operator->();
  if(debug) dbe->setVerbose(1);

  syncNumTot = 0;
  syncNum = 0;

}



// destructor
DTDigiTask::~DTDigiTask(){
  if(debug) cout << "DTDigiTask: analyzed " << nevents << " events" << endl;

}




void DTDigiTask::endJob(){
  if(debug) cout<<"[DTDigiTask] endjob called!"<<endl;
  
  dbe->rmdir("DT/Digi");
}




void DTDigiTask::beginJob(const edm::EventSetup& context){
  if(debug) cout<<"[DTDigiTask]: BeginJob"<<endl;

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // tTrig 
  if (readTTrigDB) 
    context.get<DTTtrigRcd>().get(tTrigMap);
  // t0s 
  if (subtractT0) 
    context.get<DTT0Rcd>().get(t0Map);
  // FIXME: tMax (not yet from the DB)
  tMax = defaultTmax;
  
  // ----------------------------------------------------------------------
  if(doStaticBooking) {  // Static histo booking
    for(int wh = -2; wh <= 2; ++wh) { // loop over wheels
      bookHistos(wh, string("Occupancies"), "OccupancyAllHits");
      for(int st = 1; st <= 4; ++st) { // loop over stations
	for(int sect = 1; sect <= 14; ++sect) { // loop over sectors
	  if((sect == 13 || sect == 14) && st != 4) continue;
	  // Get the chamber ID
	  const  DTChamberId dtChId(wh,st,sect);

	  // Occupancies 
	  if (!readTTrigDB) {
	    bookHistos(dtChId, string("Occupancies"), "OccupancyAllHits_perCh");
	  } else {
	    bookHistos(dtChId, string("Occupancies"), "OccupancyNoise_perCh");
	    bookHistos(dtChId, string("Occupancies"), "OccupancyInTimeHits_perCh" );
	  }

	  for(int sl = 1; sl <= 3; ++sl) { // Loop over SLs
	    if(st == 4 && sl == 2) continue;
	    const  DTSuperLayerId dtSLId(wh,st,sect,sl);
	    if(isLocalRun) {
	      bookHistos(dtSLId, string("TimeBoxes"), "TimeBox");
	    } else {
	      // TimeBoxes for different triggers
	      bookHistos(dtSLId, string("TimeBoxes"), "TimeBoxDTonly");
	      bookHistos(dtSLId, string("TimeBoxes"), "TimeBoxNoDT");
	      bookHistos(dtSLId, string("TimeBoxes"), "TimeBoxDTalso");
	    }
	  }
	}
      }
    }
  }
}




void DTDigiTask::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  if(debug) cout<<"[DTDigiTask]: Begin of LS transition"<<endl;
  
  // Reset the MonitorElements every n (= ResetCycle) Lumi Blocks
  if(lumiSeg.id().luminosityBlock() % resetCycle == 0) {
    if(debug)
      cout<<"[DTDigiTask]: Reset at the LS transition : "<<lumiSeg.id().luminosityBlock()<<endl;
    // Loop over all ME
    for(map<string, map<uint32_t, MonitorElement*> > ::const_iterator histo = digiHistos.begin();
	histo != digiHistos.end(); histo++) {
      for(map<uint32_t, MonitorElement*> ::const_iterator ht = (*histo).second.begin();
	  ht != (*histo).second.end(); ht++) {
	(*ht).second->Reset();
      }
    }
  }
  // loop over wheel summaries
  for(map<string, map<int, MonitorElement*> > ::const_iterator histos = wheelHistos.begin();
      histos != wheelHistos.end(); ++histos) {
    for(map<int, MonitorElement*>::const_iterator histo = (*histos).second.begin();
	histo != (*histos).second.end(); ++histo) {
      (*histo).second->Reset();
    }
  }

  
}




void DTDigiTask::bookHistos(const DTSuperLayerId& dtSL, string folder, string histoTag) {
  // set the folder
  stringstream wheel; wheel << dtSL.wheel();	
  stringstream station; station << dtSL.station();	
  stringstream sector; sector << dtSL.sector();	
  stringstream superLayer; superLayer << dtSL.superlayer();
  dbe->setCurrentFolder("DT/Digi/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/" + folder);

  // Build the histo name
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str(); 


  if (debug) {
    cout<<"[DTDigiTask]: booking SL histo:"<<endl;
    cout<<"              folder "<< "DT/Digi/Wheel" + wheel.str() +
      "/Station" + station.str() +
      "/Sector" + sector.str() + "/" + folder<<endl;
    cout<<"              histoTag "<<histoTag<<endl;
    cout<<"              histoName "<<histoName<<endl;
  }

  if ( readTTrigDB ) 
    tTrigMap->slTtrig( dtSL, tTrig, tTrigRMS); 
  else tTrig = defaultTTrig;
  

  if ( folder == "TimeBoxes") {
    string histoTitle = histoName + " (TDC Counts)";

    if (!readTTrigDB) {
      int maxTDCCounts = 6400 * tdcRescale;
      (digiHistos[histoTag])[dtSL.rawId()] = 
	dbe->book1D(histoName,histoTitle, maxTDCCounts/timeBoxGranularity, 0, maxTDCCounts);
    }    
    else {
      (digiHistos[histoTag])[dtSL.rawId()] = 
	dbe->book1D(histoName,histoTitle, 2*tMax/timeBoxGranularity, tTrig-tMax, tTrig+2*tMax);
      // FIXME: this is not setting the right bin size ?
    }
  }

  if ( folder == "CathodPhotoPeaks" ) 
    (digiHistos[histoTag])[dtSL.rawId()] = dbe->book1D(histoName,histoName,500,0,1000);

  
}




void DTDigiTask::bookHistos(const DTChamberId& dtCh, string folder, string histoTag) {
  // set the current folder
  stringstream wheel; wheel << dtCh.wheel();	
  stringstream station; station << dtCh.station();	
  stringstream sector; sector << dtCh.sector();
  dbe->setCurrentFolder("DT/Digi/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/" + folder);

  // build the histo name
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 

  if (debug){
    cout<<"[DTDigiTask]: booking chamber histo:"<<endl;
    cout<<"              folder "<< "DT/Digi/Wheel" + wheel.str() +
      "/Station" + station.str() +
      "/Sector" + sector.str() + "/" + folder<<endl;
    cout<<"              histoTag "<<histoTag<<endl;
    cout<<"              histoName "<<histoName<<endl;
  }
  
  if (folder == "Occupancies")    {
    
    const DTChamber* dtchamber = muonGeom->chamber(dtCh);
    const std::vector<const DTSuperLayer*> dtSupLylist = dtchamber->superLayers();
    std::vector<const DTSuperLayer*>::const_iterator suly = dtSupLylist.begin();
    std::vector<const DTSuperLayer*>::const_iterator sulyend = dtSupLylist.end();
    
    int nWires = 0;
    int firstWire = 0;
    int nWires_max = 0;
    
    while(suly != sulyend) {
      const std::vector<const DTLayer*> dtLyList = (*suly)->layers();
      std::vector<const DTLayer*>::const_iterator ly = dtLyList.begin();
      std::vector<const DTLayer*>::const_iterator lyend = dtLyList.end();
      stringstream superLayer; superLayer << (*suly)->id().superlayer();
      
      while(ly != lyend) {
	nWires = muonGeom->layer((*ly)->id())->specificTopology().channels();
	firstWire = muonGeom->layer((*ly)->id())->specificTopology().firstChannel();
	stringstream layer; layer << (*ly)->id().layer();
	string histoName_layer = histoName + "_SL" + superLayer.str()  + "_L" + layer.str();
	if(histoTag == "OccupancyAllHits_perL" 
	   || histoTag == "OccupancyNoise_perL"
	   || histoTag == "OccupancyInTimeHits_perL")
	  (digiHistos[histoTag])[(*ly)->id().rawId()] = dbe->book1D(histoName_layer,histoName_layer,nWires,firstWire,nWires+firstWire);
	++ly;
	if((nWires+firstWire) > nWires_max) nWires_max = (nWires+firstWire);
	
      }
      ++suly;
    }
   
    if(histoTag != "OccupancyAllHits_perL" 
	   && histoTag != "OccupancyNoise_perL"
	   && histoTag != "OccupancyInTimeHits_perL"){
      (digiHistos[histoTag])[dtCh.rawId()] = dbe->book2D(histoName,histoName,nWires_max,1,nWires_max+1,12,0,12);
    
      for(int i=1;i<=12;i++) { 
	if(i<5){
	  stringstream layer;
	  string layer_name;
	  layer<<i;
	  layer>>layer_name;
	  string label="SL1: L"+layer_name;
	  (digiHistos[histoTag])[dtCh.rawId()]->setBinLabel(i,label,2);
	}
	else if(i>4 && i<9){
	  stringstream layer;
	  string layer_name;
	  layer<<(i-4);
	  layer>>layer_name;
	  string label="SL2: L"+layer_name;
	  (digiHistos[histoTag])[dtCh.rawId()]->setBinLabel(i,label,2);
	}
	else if(i>8 && i<13){
	  stringstream layer;
	  string layer_name;
	  layer<<(i-8);
	  layer>>layer_name;
	  string label="SL3: L"+layer_name;
	  (digiHistos[histoTag])[dtCh.rawId()]->setBinLabel(i,label,2);
	}
      }
    }
  }
}




void DTDigiTask::bookHistos(const int wheelId, string folder, string histoTag) {
  // Set the current folder
  stringstream wheel; wheel << wheelId;	
  dbe->setCurrentFolder("DT/Digi/Wheel" + wheel.str());

  // build the histo name
  string histoName = histoTag + "_W" + wheel.str(); 
  
  if(debug) {
    cout<<"[DTDigiTask]: booking wheel histo:"<<endl;
    cout<<"              folder "<< "DT/Digi/Wheel" + wheel.str() + "/" + folder<<endl;
    cout<<"              histoTag "<<histoTag<<endl;
    cout<<"              histoName "<<histoName<<endl;
  }
  
  string histoTitle = "# of digis per chamber WHEEL: "+wheel.str();
  (wheelHistos[histoTag])[wheelId] = dbe->book2D(histoName,histoTitle,12,1,13,4,1,5);
  (wheelHistos[histoTag])[wheelId]->setBinLabel(1,"MB1",2);
  (wheelHistos[histoTag])[wheelId]->setBinLabel(2,"MB2",2);
  (wheelHistos[histoTag])[wheelId]->setBinLabel(3,"MB3",2);
  (wheelHistos[histoTag])[wheelId]->setBinLabel(4,"MB4",2);
  (wheelHistos[histoTag])[wheelId]->setAxisTitle("sector",1);
}



// does the real job
void DTDigiTask::analyze(const edm::Event& event, const edm::EventSetup& c) {
  if(debug) cout << "[DTDigiTask] analyze" << endl;
  
  nevents++;

  if (debug && nevents%1000 == 0) {
    cout << "[DTDigiTask] Analyze #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;
  }
  
  // Get the ingredients from the event
  
  // Digi collection
  edm::Handle<DTDigiCollection> dtdigis;
  event.getByLabel(dtDigiLabel, dtdigis);

  // LTC digis
  if (!isLocalRun) event.getByType(ltcdigis);

  // Status map (for noisy channels)
  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    if(debug) cout << "    get the map of noisy channels" << endl;
    // Get the map of noisy channels
    c.get<DTStatusFlagRcd>().get(statusMap);
  }

  string histoTag;

  int tdcCount = 0;

  bool doSync = false; //FIXME: switch off but don't exactly understand what it's doing

  // Check if the digi container is empty
  if(dtdigis->begin() == dtdigis->end()) {
    doSync = false;
    if(debug) cout << "Event " << nevents << " empty." << endl;
  }

  if (doSync) { // dosync
    if(debug) cout << "     doSync" << endl;
    DTChamberId chDone = ((*(dtdigis->begin())).first).chamberId();
    
    DTDigiCollection::DigiRangeIterator dtLayerId_It;
    for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); dtLayerId_It++){
      DTChamberId chId = ((*dtLayerId_It).first).chamberId();
      if (!(chId == chDone)) {
	hitMap.insert(make_pair(chDone,tdcCount));
	tdcCount = 0;
	chDone = chId;

      }
      
      
      tdcCount += (((*dtLayerId_It).second).second - ((*dtLayerId_It).second).first);
            
    }
    
    hitMap.insert(make_pair(chDone,tdcCount));
    
    bool eventSync = false;
    bool stat = false;
    std::map<DTChamberId,int>::iterator iter;
    for (iter = hitMap.begin(); iter != hitMap.end(); iter++) {
      
      stat = false;
      if ((iter->second) > maxTDCHits) { 
	stat = true;
	eventSync = true;
      }
      hitMapCheck.insert(make_pair((iter->first),stat));

    }
    hitMap.clear();
    
    if (eventSync) {
      cout << "Event " << nevents << " probably sync noisy: time box not filled! " << endl;
      syncNumTot++;
      syncNum++;
    }
    if (nevents%1000 == 0) {
      cout << (syncNumTot*100./nevents) << "% sync noise events since the beginning " << endl;
      cout << (syncNum*0.1) << "% sync noise events in the last 1000 events " << endl;
      syncNum = 0;
    }
  }

  bool isSyncNoisy;

  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It) { // Loop over layers


    isSyncNoisy = false;
    if (doSync) {
      DTChamberId chId = ((*dtLayerId_It).first).superlayerId().chamberId();
      std::map<DTChamberId,bool>::iterator iterch;
      
      for (iterch = hitMapCheck.begin(); iterch != hitMapCheck.end(); iterch++) {
	if ((iterch->first) == chId) isSyncNoisy = iterch->second;

      }

    }

    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt) { // Loop over all digis

	bool isNoisy = false;
	bool isFEMasked = false;
	bool isTDCMasked = false;
	bool isTrigMask = false;
	bool isDead = false;
	bool isNohv = false;
	if(checkNoisyChannels) {
	  const DTWireId wireId(((*dtLayerId_It).first), (*digiIt).wire());
	  statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
	}
	
	

	// Get the useful IDs
	const  DTSuperLayerId dtSLId = ((*dtLayerId_It).first).superlayerId();
	uint32_t indexSL = dtSLId.rawId();
	const  DTChamberId dtChId = dtSLId.chamberId(); 
	uint32_t indexCh = dtChId.rawId();
	int layer_number=((*dtLayerId_It).first).layer();
	int superlayer_number=dtSLId.superlayer();
	const  DTLayerId dtLId = (*dtLayerId_It).first;
	
	// Read the ttrig DB or set a rough value from config
	if (readTTrigDB)
	  tTrigMap->slTtrig( ((*dtLayerId_It).first).superlayerId(), tTrig, tTrigRMS); 
	else tTrig = defaultTTrig;
	
	int inTimeHitsLowerBoundCorr = int(round(tTrig)) - inTimeHitsLowerBound;
	int inTimeHitsUpperBoundCorr = int(round(tTrig)) + tMax + inTimeHitsUpperBound;
	
	float t0; float t0RMS;
	int tdcTime = (*digiIt).countsTDC();
	
	if (subtractT0) {
	  const DTWireId dtWireId(((*dtLayerId_It).first), (*digiIt).wire());
	  t0Map->cellT0(dtWireId, t0, t0RMS) ;
	  tdcTime += int(round(t0));
	}

	

	// Fill Time-Boxes
	// NOTE: avoid to fill TB and PhotoPeak with noise. Occupancy are filled anyway
	if (( !isNoisy ) && (!isSyncNoisy)) { // Discard noisy channels
	  // TimeBoxes per SL
	  histoTag = "TimeBox" + triggerSource();
	  if (digiHistos[histoTag].find(indexSL) == digiHistos[histoTag].end())
	    bookHistos( dtSLId, string("TimeBoxes"), histoTag );
	  (digiHistos.find(histoTag)->second).find(indexSL)->second->Fill(tdcTime);
	  
	  // FIXME: remove the time distribution for the after-pulses	  
	  // 2nd - 1st (CathodPhotoPeak) per SL
	  // 	  if ( (*digiIt).number() == 1 ) {
	    
	  // 	    DTDigiCollection::const_iterator firstDigiIt = digiIt;
	  // 	    firstDigiIt--;
	    
	  // 	    histoTag = "CathodPhotoPeak";
	  // 	    if (digiHistos[histoTag].find(indexSL) == digiHistos[histoTag].end())
	  // 	      bookHistos( dtSLId, string("CathodPhotoPeaks"), histoTag );
	  // 	    (digiHistos.find(histoTag)->second).find(indexSL)->second->Fill((*digiIt).countsTDC()-
	  // 									    (*firstDigiIt).countsTDC());
	  // 	  }
	}

	// Fill Occupancies
	if (!isSyncNoisy) { // Discard synch noisy channels 
	  if (!readTTrigDB) { // Do not use ttrig table
	    //Occupancies per chamber & layer
	    histoTag = "OccupancyAllHits_perCh";
	    map<uint32_t, MonitorElement*>::const_iterator mappedHisto =
	      digiHistos[histoTag].find(indexCh);
	    if (mappedHisto == digiHistos[histoTag].end()) { // dynamic booking
	      bookHistos(dtChId, string("Occupancies"), histoTag);
	      mappedHisto = digiHistos[histoTag].find(indexCh);
	    }
	    mappedHisto->second->Fill((*digiIt).wire(),(layer_number+(superlayer_number-1)*4)-1);

	    
	    // Fill the chamber occupancy
	    histoTag = "OccupancyAllHits";
	    map<int, MonitorElement*>::const_iterator histoPerWheel =
	      wheelHistos[histoTag].find(dtChId.wheel());
	    if(histoPerWheel ==  wheelHistos[histoTag].end()) { // dynamic booking
	      bookHistos(dtChId.wheel(), string("Occupancies"), histoTag);
	      histoPerWheel = wheelHistos[histoTag].find(dtChId.wheel());
	    }
	    histoPerWheel->second->Fill(dtChId.sector(),dtChId.station()); // FIXME: normalize to # of layers
	   
	    
	  } else { // after-Calibration jobs: use ttrig DB

	    if (tdcTime < inTimeHitsLowerBoundCorr ) { // FIXME: what about tdcTime > inTimeHitsUpperBoundCorr ???
	      // Noise: Before tTrig
	      
	      //Occupancies Noise per chamber & layer
	      histoTag = "OccupancyNoise_perCh";
	      map<uint32_t, MonitorElement*>::const_iterator mappedHisto =
		digiHistos[histoTag].find(indexCh);
	      if(mappedHisto == digiHistos[histoTag].end()) {
		bookHistos(dtChId, string("Occupancies"), histoTag);
		mappedHisto = digiHistos[histoTag].find(indexCh);
	      }
	      mappedHisto->second->Fill((*digiIt).wire(),
					(layer_number+(superlayer_number-1)*4)-1);

	    } else if (tdcTime > inTimeHitsLowerBoundCorr && tdcTime < inTimeHitsUpperBoundCorr) { 
	      // Physical hits: within the time window	
	      
	      //Occupancies Signal per chamber & layer
	      histoTag = "OccupancyInTimeHits_perCh";
	      map<uint32_t, MonitorElement*>::const_iterator mappedHisto =
		digiHistos[histoTag].find(indexCh);
	      if(mappedHisto == digiHistos[histoTag].end()) {
		bookHistos(dtChId, string("Occupancies"), histoTag);
		mappedHisto = digiHistos[histoTag].find(indexCh);
	      }
	      mappedHisto->second->Fill((*digiIt).wire(),
					(layer_number+(superlayer_number-1)*4)-1);

	    }
	  }
	}
    }
  }
  
  hitMapCheck.clear();
}


string DTDigiTask::triggerSource() {

  string l1ASource;

  if (!isLocalRun) {
    for (std::vector<LTCDigi>::const_iterator ltc_it = ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
      int otherTriggerSum=0;
      for (int i = 1; i < 6; i++)
	otherTriggerSum += int((*ltc_it).HasTriggered(i));
      
      if ((*ltc_it).HasTriggered(0) && otherTriggerSum == 0) 
	l1ASource = "DTonly";
      else if (!(*ltc_it).HasTriggered(0))
	l1ASource = "NoDT";
      else if ((*ltc_it).HasTriggered(0) && otherTriggerSum > 0)
	l1ASource = "DTalso";
    }
  }

  return l1ASource;

}








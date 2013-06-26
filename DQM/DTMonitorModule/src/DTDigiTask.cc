/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2012/09/24 16:08:06 $
 * $Revision: 1.70 $
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
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

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
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;



// Contructor
DTDigiTask::DTDigiTask(const edm::ParameterSet& ps){
  // switch for the verbosity
  LogTrace("DTDQM|DTMonitorModule|DTDigiTask") << "[DTDigiTask]: Constructor" << endl;

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
  if (!isLocalRun) {
    ltcDigiCollectionTag = ps.getParameter<edm::InputTag>("ltcDigiCollectionTag");
  }

  // Setting for the reset of the ME after n (= ResetCycle) luminosity sections
  resetCycle = ps.getUntrackedParameter<int>("ResetCycle", 3);
  // Check the DB of noisy channels
  checkNoisyChannels = ps.getUntrackedParameter<bool>("checkNoisyChannels",false);
  // Default TTrig to be used when not reading the TTrig DB
  defaultTTrig = ps.getParameter<int>("defaultTtrig");
  inTimeHitsLowerBound = ps.getParameter<int>("inTimeHitsLowerBound");
  inTimeHitsUpperBound = ps.getParameter<int>("inTimeHitsUpperBound");
  timeBoxGranularity = ps.getUntrackedParameter<int>("timeBoxGranularity",4);
  maxTDCCounts = ps.getUntrackedParameter<int>("maxTDCCounts", 6400);

  doAllHitsOccupancies = ps.getUntrackedParameter<bool>("doAllHitsOccupancies", true);
  doNoiseOccupancies = ps.getUntrackedParameter<bool>("doNoiseOccupancies", false);
  doInTimeOccupancies = ps.getUntrackedParameter<bool>("doInTimeOccupancies", false);

  // switch on the mode for running on test pulses (different top folder)
  tpMode = ps.getUntrackedParameter<bool>("testPulseMode", false);
  // switch on/off the filtering of synchronous noise events (cutting on the # of digis)
  // time-boxes and occupancy plots are not filled and summary plots are created to report the problem
  filterSyncNoise = ps.getUntrackedParameter<bool>("filterSyncNoise", false);
  // look for synch noisy events, produce histograms but do not filter them
  lookForSyncNoise = ps.getUntrackedParameter<bool>("lookForSyncNoise", false);
  // switch on production of time-boxes with layer granularity
  doLayerTimeBoxes = ps.getUntrackedParameter<bool>("doLayerTimeBoxes", false);

  dbe = edm::Service<DQMStore>().operator->();

  syncNumTot = 0;
  syncNum = 0;

}



// destructor
DTDigiTask::~DTDigiTask(){
  LogTrace("DTDQM|DTMonitorModule|DTDigiTask") << "DTDigiTask: analyzed " << nevents << " events" << endl;

}




void DTDigiTask::endJob(){
  LogTrace("DTDQM|DTMonitorModule|DTDigiTask") <<"[DTDigiTask] endjob called!"<<endl;

}




void DTDigiTask::beginJob(){
  LogTrace("DTDQM|DTMonitorModule|DTDigiTask") <<"[DTDigiTask]: BeginJob"<<endl;

  nevents = 0;
}


void DTDigiTask::beginRun(const edm::Run& run, const edm::EventSetup& context) {
  LogTrace("DTDQM|DTMonitorModule|DTDigiTask") << "[DTDigiTask]: begin run" << endl;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // map of the channels
  context.get<DTReadOutMappingRcd>().get(mapping);

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
    // book the event counter
    dbe->setCurrentFolder("DT/EventInfo/Counters");
    nEventMonitor = dbe->bookFloat(tpMode ? "nProcessedEventsDigiTP" : "nProcessedEventsDigi" );
    dbe->setCurrentFolder(topFolder());
    for(int wh = -2; wh <= 2; ++wh) { // loop over wheels
      if(doAllHitsOccupancies) bookHistos(wh,string("Occupancies"),"OccupancyAllHits");
      if(doNoiseOccupancies) bookHistos(wh,string("Occupancies"),"OccupancyNoiseHits");
      if(doInTimeOccupancies) bookHistos(wh,string("Occupancies"),"OccupancyInTimeHits");

      if(lookForSyncNoise || filterSyncNoise) {
        bookHistos(wh,string("SynchNoise"),"SyncNoiseEvents");
        bookHistos(wh,string("SynchNoise"),"SyncNoiseChambs");
      }

      for(int st = 1; st <= 4; ++st) { // loop over stations
        for(int sect = 1; sect <= 14; ++sect) { // loop over sectors
          if((sect == 13 || sect == 14) && st != 4) continue;
          // Get the chamber ID
          const  DTChamberId dtChId(wh,st,sect);

          // Occupancies 
          if (doAllHitsOccupancies) { 
            bookHistos(dtChId,string("Occupancies"),"OccupancyAllHits_perCh");
            // set channel mapping
            channelsMap(dtChId, "OccupancyAllHits_perCh");
          }
          if(doNoiseOccupancies) 
            bookHistos(dtChId,string("Occupancies"),"OccupancyNoise_perCh");
          if(doInTimeOccupancies)
            bookHistos(dtChId,string("Occupancies"),"OccupancyInTimeHits_perCh");




          for(int sl = 1; sl <= 3; ++sl) { // Loop over SLs
            if(st == 4 && sl == 2) continue;
            const  DTSuperLayerId dtSLId(wh,st,sect,sl);
            if(isLocalRun) {
              bookHistos(dtSLId,string("TimeBoxes"),"TimeBox");
            } else {
              // TimeBoxes for different triggers
              bookHistos(dtSLId,string("TimeBoxes"),"TimeBoxDTonly");
              bookHistos(dtSLId,string("TimeBoxes"),"TimeBoxNoDT");
              bookHistos(dtSLId,string("TimeBoxes"),"TimeBoxDTalso");
            }
          }
        }
      }
    }
  }
}




void DTDigiTask::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  LogTrace("DTDQM|DTMonitorModule|DTDigiTask") << "[DTDigiTask]: Begin of LS transition" << endl;

  // Reset the MonitorElements every n (= ResetCycle) Lumi Blocks
  int lumiBlock = lumiSeg.id().luminosityBlock(); 
  if(lumiBlock % resetCycle == 0) {
    LogVerbatim("DTDQM|DTMonitorModule|DTDigiTask")
      <<"[DTDigiTask]: Reset at the LS transition : " 
      << lumiBlock << endl;
    // Loop over all ME
    map<string,map<uint32_t,MonitorElement*> >::const_iterator histosIt  = digiHistos.begin();
    map<string,map<uint32_t,MonitorElement*> >::const_iterator histosEnd = digiHistos.end();
    for(;histosIt != histosEnd ; ++histosIt) {
      map<uint32_t,MonitorElement*>::const_iterator histoIt  = (*histosIt).second.begin();
      map<uint32_t,MonitorElement*>::const_iterator histoEnd = (*histosIt).second.end();
      for(;histoIt != histoEnd; ++histoIt) { (*histoIt).second->Reset(); }
    }

    // re-set mapping for not real channels in the occupancyHits per chamber
    for(int wh=-2; wh<=2; wh++) {
      for(int sect=1; sect<=14; sect++) {
        for(int st=1; st<=4; st++) {
          if( (sect == 13 || sect == 14) && st != 4 ) {continue;}
           const DTChamberId dtChId(wh,st,sect);
           channelsMap(dtChId, "OccupancyAllHits_perCh");
        }
      }
    }

    // loop over wheel summaries
    map<string,map<int,MonitorElement*> >::const_iterator whHistosIt  = wheelHistos.begin();
    map<string,map<int,MonitorElement*> >::const_iterator whHistosEnd = wheelHistos.end();
    for(; whHistosIt != whHistosEnd ; ++whHistosIt) {
      if ((*whHistosIt).first.find("Sync") == string::npos) { // FIXME skips synch noise plots
        map<int,MonitorElement*>::const_iterator histoIt  = (*whHistosIt).second.begin();
        map<int,MonitorElement*>::const_iterator histoEnd = (*whHistosIt).second.end();
        for(;histoIt != histoEnd; ++histoIt) { (*histoIt).second->Reset(); }
      }
    }
  }

}




void DTDigiTask::bookHistos(const DTSuperLayerId& dtSL, string folder, string histoTag) {
  // set the folder
  stringstream wheel; wheel << dtSL.wheel();	
  stringstream station; station << dtSL.station();	
  stringstream sector; sector << dtSL.sector();	
  stringstream superLayer; superLayer << dtSL.superlayer();
  dbe->setCurrentFolder(topFolder() + "Wheel" + wheel.str() +
      "/Sector" + sector.str() +
      "/Station" + station.str());

  // Build the histo name
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str(); 

  LogTrace("DTDQM|DTMonitorModule|DTDigiTask")
    << "[DTDigiTask]: booking SL histo:" << histoName
    << " (tag: " << histoTag
    << ") folder: " << topFolder() + "Wheel" + wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/" + folder << endl;


  // ttrig and rms are TDC counts
  if ( readTTrigDB ) 
    tTrigMap->get(dtSL, tTrig, tTrigRMS, kFactor, DTTimeUnits::counts); 
  else tTrig = defaultTTrig;


  if ( folder == "TimeBoxes") {
    string histoTitle = histoName + " (TDC Counts)";

    if (!readTTrigDB) {
      (digiHistos[histoTag])[dtSL.rawId()] = 
        dbe->book1D(histoName,histoTitle, maxTDCCounts/timeBoxGranularity, 0, maxTDCCounts);
      if(doLayerTimeBoxes) {      // Book TimeBoxes per layer
        for(int layer = 1; layer != 5; ++layer) {
          DTLayerId layerId(dtSL, layer);
          stringstream layerHistoName; layerHistoName << histoName << "_L" << layer;
          (digiHistos[histoTag])[layerId.rawId()] =
            dbe->book1D(layerHistoName.str(),layerHistoName.str(), maxTDCCounts/timeBoxGranularity, 0, maxTDCCounts);
        }
      }
    }    
    else {
      (digiHistos[histoTag])[dtSL.rawId()] = 
        dbe->book1D(histoName,histoTitle, 3*tMax/timeBoxGranularity, tTrig-tMax, tTrig+2*tMax);
      if(doLayerTimeBoxes) {
        // Book TimeBoxes per layer
        for(int layer = 1; layer != 5; ++layer) {
          DTLayerId layerId(dtSL, layer);
          stringstream layerHistoName; layerHistoName << histoName << "_L" << layer;
          (digiHistos[histoTag])[layerId.rawId()] =
            dbe->book1D(layerHistoName.str(),layerHistoName.str(), 3*tMax/timeBoxGranularity, tTrig-tMax, tTrig+2*tMax);
        }
      }
    }
  }

  if ( folder == "CathodPhotoPeaks" ) {
    dbe->setCurrentFolder(topFolder() + "Wheel" + wheel.str() +
        "/Sector" + sector.str() + 
        "/Station" + station.str() + "/" + folder);
    (digiHistos[histoTag])[dtSL.rawId()] = dbe->book1D(histoName,histoName,500,0,1000);
  }

}




void DTDigiTask::bookHistos(const DTChamberId& dtCh, string folder, string histoTag) {
  // set the current folder
  stringstream wheel; wheel << dtCh.wheel();	
  stringstream station; station << dtCh.station();	
  stringstream sector; sector << dtCh.sector();
  dbe->setCurrentFolder(topFolder() + "Wheel" + wheel.str() +
      "/Sector" + sector.str() + 
      "/Station" + station.str());

  // build the histo name
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 


  LogTrace("DTDQM|DTMonitorModule|DTDigiTask")
    << "[DTDigiTask]: booking chamber histo:" 
    << " (tag: " << histoTag
    << ") folder: " << topFolder() + "Wheel" + wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() << endl;


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
      // Set the title to show the time interval used (only if unique == not from DB)
      string histoTitle = histoName;
      if(!readTTrigDB && histoTag == "OccupancyInTimeHits_perCh") {
        stringstream title;
        int inTimeHitsLowerBoundCorr = int(round(defaultTTrig)) - inTimeHitsLowerBound;
        int inTimeHitsUpperBoundCorr = int(round(defaultTTrig)) + defaultTmax + inTimeHitsUpperBound;
        title << "Occ. digis in time [" << inTimeHitsLowerBoundCorr << ", "
          << inTimeHitsUpperBoundCorr << "] (TDC counts)";
        histoTitle = title.str();
      }
      (digiHistos[histoTag])[dtCh.rawId()] = dbe->book2D(histoName,histoTitle,nWires_max,1,nWires_max+1,12,0,12);

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


  // build the histo name
  string histoName = histoTag + "_W" + wheel.str(); 


  LogTrace("DTDQM|DTMonitorModule|DTDigiTask")
    << "[DTDigiTask]: booking wheel histo:" << histoName
    << " (tag: " << histoTag
    << ") folder: " << topFolder() + "Wheel" + wheel.str() + "/" <<endl;

  if(folder == "Occupancies") {
    dbe->setCurrentFolder(topFolder() + "Wheel" + wheel.str());
    string histoTitle = "# of digis per chamber WHEEL: "+wheel.str();
    (wheelHistos[histoTag])[wheelId] = dbe->book2D(histoName,histoTitle,12,1,13,4,1,5);
    (wheelHistos[histoTag])[wheelId]->setBinLabel(1,"MB1",2);
    (wheelHistos[histoTag])[wheelId]->setBinLabel(2,"MB2",2);
    (wheelHistos[histoTag])[wheelId]->setBinLabel(3,"MB3",2);
    (wheelHistos[histoTag])[wheelId]->setBinLabel(4,"MB4",2);
    (wheelHistos[histoTag])[wheelId]->setAxisTitle("sector",1);
  } else if(folder == "SynchNoise") {
    dbe->setCurrentFolder("DT/05-Noise/SynchNoise");
    if (histoTag== "SyncNoiseEvents") {
      string histoTitle = "# of Syncronous-noise events WHEEL: "+wheel.str();
      (wheelHistos[histoTag])[wheelId] = dbe->book2D(histoName,histoTitle,12,1,13,4,1,5);
      (wheelHistos[histoTag])[wheelId]->setBinLabel(1,"MB1",2);
      (wheelHistos[histoTag])[wheelId]->setBinLabel(2,"MB2",2);
      (wheelHistos[histoTag])[wheelId]->setBinLabel(3,"MB3",2);
      (wheelHistos[histoTag])[wheelId]->setBinLabel(4,"MB4",2);
      (wheelHistos[histoTag])[wheelId]->setAxisTitle("sector",1);
    } else if (histoTag== "SyncNoiseChambs") {
      string histoTitle = "# of Synchornous-noise chamb per evt. WHEEL: "+wheel.str();
      (wheelHistos[histoTag])[wheelId] = dbe->book1D(histoName,histoTitle,50,0.5,50.5);
      (wheelHistos[histoTag])[wheelId]->setAxisTitle("# of noisy chambs.",1);
      (wheelHistos[histoTag])[wheelId]->setAxisTitle("# of evts.",2);
    }
  }

}



// does the real job
void DTDigiTask::analyze(const edm::Event& event, const edm::EventSetup& c) {
  nevents++;
  nEventMonitor->Fill(nevents);
  if (nevents%1000 == 0) {
    LogTrace("DTDQM|DTMonitorModule|DTDigiTask") << "[DTDigiTask] Analyze #Run: " << event.id().run()
      << " #Event: " << event.id().event() << endl;
  }

  // Get the ingredients from the event

  // Digi collection
  edm::Handle<DTDigiCollection> dtdigis;
  event.getByLabel(dtDigiLabel, dtdigis);

  // LTC digis
  if (!isLocalRun) event.getByLabel(ltcDigiCollectionTag, ltcdigis);

  // Status map (for noisy channels)
  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    // Get the map of noisy channels
    c.get<DTStatusFlagRcd>().get(statusMap);
  }

  string histoTag;


  // Check if the digi container is empty
  if(dtdigis->begin() == dtdigis->end()) {
    LogTrace("DTDQM|DTMonitorModule|DTDigiTask") << "Event " << nevents << " empty." << endl;
  }

  if (lookForSyncNoise || filterSyncNoise) { // dosync
    // Count the # of digis per chamber
    DTDigiCollection::DigiRangeIterator dtLayerId_It;
    for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); dtLayerId_It++) {
      DTChamberId chId = ((*dtLayerId_It).first).chamberId();
      if(hitMap.find(chId) == hitMap.end()) {// new chamber
        hitMap[chId] = 0;
      }
      hitMap[chId] += (((*dtLayerId_It).second).second - ((*dtLayerId_It).second).first);
    }



    // check chamber with # of digis above threshold and flag them as noisy
    map<DTChamberId,int>::const_iterator hitMapIt  = hitMap.begin();
    map<DTChamberId,int>::const_iterator hitMapEnd = hitMap.end();

    map<int,int> chMap;

    for (; hitMapIt != hitMapEnd; ++hitMapIt) {
      if((hitMapIt->second) > maxTDCHits) { 

        DTChamberId chId = hitMapIt->first;
        int wh = chId.wheel();

        LogTrace("DTDQM|DTMonitorModule|DTDigiTask") << "[DTDigiTask] Synch noise in chamber: " << chId
          << " with # digis: " << hitMapIt->second << endl;

        if(chMap.find(wh) == chMap.end()) { chMap[wh] = 0; }
        chMap[wh]++ ;

        syncNoisyChambers.insert(chId);

        wheelHistos["SyncNoiseEvents"][wh]->Fill(chId.sector(),chId.station()); 

        // Only needed in case of ratio map not used right now
        // FIXME check and optimize
        // nSynchNoiseEvents[iter->first]++;	
        // FIXME: should update all chambers each event 
        // wheelHistos["SyncNoiseEvents"][(*iter).first.wheel()]->setBinContent((*iter).first.sector(),(*iter).first.station(),
        // 								(double)nSynchNoiseEvents[iter->first]/(double)nevents); 

      }
    }

    // fill # of noisy ch per wheel plot
    map<int,int>::const_iterator chMapIt  = chMap.begin();
    map<int,int>::const_iterator chMapEnd = chMap.end();
    for (; chMapIt != chMapEnd; ++chMapIt) { 
      wheelHistos["SyncNoiseChambs"][(*chMapIt).first]->Fill((*chMapIt).second); 
    }   

    // clear the map of # of digis per chamber: not needed anymore
    hitMap.clear();

    if (syncNoisyChambers.size() != 0) {
      LogVerbatim("DTDQM|DTMonitorModule|DTDigiTask") << "[DTDigiTask] Synch Noise in event: " << nevents;
      if(filterSyncNoise) LogVerbatim("DTDQM|DTMonitorModule|DTDigiTask") << "\tnoisy time-boxes and occupancy will not be filled!" << endl; 
      syncNumTot++;
      syncNum++;
    }

    // Logging of "large" synch Noisy events in private DQM
    if (syncNoisyChambers.size() > 3) {
      time_t eventTime = time_t(event.time().value()>>32);

      LogVerbatim("DTDQM|DTMonitorModule|DTDigiTask|DTSynchNoise") 
        << "[DTDigiTask] At least 4 Synch Noisy chambers in Run : " << event.id().run() 
        << " Lumi : "  << event.id().luminosityBlock()
        << " Event : " << event.id().event()
        << " at time : " << ctime(&eventTime) << endl;

      set<DTChamberId>::const_iterator chIt  = syncNoisyChambers.begin();
      set<DTChamberId>::const_iterator chEnd = syncNoisyChambers.end();

      stringstream synchNoisyCh;
      for (;chIt!=chEnd;++chIt) { synchNoisyCh << " " << (*chIt); }
      LogVerbatim("DTDQM|DTMonitorModule|DTDigiTask|DTSynchNoise") << 
        "[DTDigiTask] Chamber List :" << synchNoisyCh.str() << endl;

    }


    if (nevents%1000 == 0) {
      LogVerbatim("DTDQM|DTMonitorModule|DTDigiTask") << (syncNumTot*100./nevents) << "% sync noise events since the beginning \n"
        << (syncNum*0.1) << "% sync noise events in the last 1000 events " << endl;
      syncNum = 0;
    }
  }

  bool isSyncNoisy = false;

  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It) { // Loop over layers
    isSyncNoisy = false;
    // check if chamber labeled as synch noisy
    if (filterSyncNoise) {
      DTChamberId chId = ((*dtLayerId_It).first).chamberId();
      if(syncNoisyChambers.find(chId) != syncNoisyChambers.end()) {
        isSyncNoisy = true;
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

      // Read the ttrig DB or set a rough value from config
      // ttrig and rms are TDC counts
      if (readTTrigDB)
        tTrigMap->get( ((*dtLayerId_It).first).superlayerId(),
            tTrig, tTrigRMS, kFactor, DTTimeUnits::counts); 
      else tTrig = defaultTTrig;

      int inTimeHitsLowerBoundCorr = int(round(tTrig)) - inTimeHitsLowerBound;
      int inTimeHitsUpperBoundCorr = int(round(tTrig)) + tMax + inTimeHitsUpperBound;

      float t0; float t0RMS;
      int tdcTime = (*digiIt).countsTDC();

      if (subtractT0) {
        const DTWireId dtWireId(((*dtLayerId_It).first), (*digiIt).wire());
        // t0s and rms are TDC counts
        t0Map->get(dtWireId, t0, t0RMS, DTTimeUnits::counts) ;
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
        if(doLayerTimeBoxes)
          (digiHistos.find(histoTag)->second).find((*dtLayerId_It).first.rawId())->second->Fill(tdcTime);
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

        if (doAllHitsOccupancies) { // fill occupancies for all hits
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


        } 

        if(doNoiseOccupancies) { // fill occupancies for hits before the ttrig
          if (tdcTime < inTimeHitsLowerBoundCorr ) { 
            // FIXME: what about tdcTime > inTimeHitsUpperBoundCorr ???

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

            // Fill the chamber occupancy
            histoTag = "OccupancyNoise";
            map<int, MonitorElement*>::const_iterator histoPerWheel =
              wheelHistos[histoTag].find(dtChId.wheel());
            if(histoPerWheel ==  wheelHistos[histoTag].end()) { // dynamic booking
              bookHistos(dtChId.wheel(), string("Occupancies"), histoTag);
              histoPerWheel = wheelHistos[histoTag].find(dtChId.wheel());
            }
            histoPerWheel->second->Fill(dtChId.sector(),dtChId.station()); // FIXME: normalize to # of layers

          } 
        }

        if(doInTimeOccupancies) { // fill occpunacies for in-time hits only
          if (tdcTime > inTimeHitsLowerBoundCorr && tdcTime < inTimeHitsUpperBoundCorr) { 
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

            // Fill the chamber occupancy
            histoTag = "OccupancyInTimeHits";
            map<int, MonitorElement*>::const_iterator histoPerWheel =
              wheelHistos[histoTag].find(dtChId.wheel());
            if(histoPerWheel ==  wheelHistos[histoTag].end()) { // dynamic booking
              bookHistos(dtChId.wheel(), string("Occupancies"), histoTag);
              histoPerWheel = wheelHistos[histoTag].find(dtChId.wheel());
            }
            histoPerWheel->second->Fill(dtChId.sector(),dtChId.station()); // FIXME: normalize to # of layers

          }
        }
      }
    }
  }

  syncNoisyChambers.clear();
}


string DTDigiTask::triggerSource() {

  string l1ASource;
  if (isLocalRun)
    return l1ASource;

  for (std::vector<LTCDigi>::const_iterator ltc_it = ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
    size_t otherTriggerSum=0;
    for (size_t i = 1; i < 6; i++)
      otherTriggerSum += size_t((*ltc_it).HasTriggered(i));

    if ((*ltc_it).HasTriggered(0) && otherTriggerSum == 0) 
      l1ASource = "DTonly";
    else if (!(*ltc_it).HasTriggered(0))
      l1ASource = "NoDT";
    else if ((*ltc_it).HasTriggered(0) && otherTriggerSum > 0)
      l1ASource = "DTalso";
  }

  return l1ASource;

}


string DTDigiTask::topFolder() const {

  if(tpMode) return string("DT/10-TestPulses/");
  return string("DT/01-Digi/");

}




void DTDigiTask::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& setup) {

  // To be used for ratio plots not used right now
  //  Update all histos for SynchNoise if needed
  //   if(lookForSyncNoise || filterSyncNoise) {
  //     //loop over chambers with synch noise events and update their entries in the histos
  //     for(map<DTChamberId, int>::const_iterator nEvPerch = nSynchNoiseEvents.begin();
  // 	nEvPerch != nSynchNoiseEvents.end(); ++nEvPerch) {
  //       DTChamberId chId = (*nEvPerch).first;
  //       wheelHistos["SyncNoiseEvents"][chId.wheel()]->setBinContent(chId.sector(),chId.station(),
  // 								  (double)nSynchNoiseEvents[chId]/(double)nevents); 
  //    }
  //  }

}

void DTDigiTask::channelsMap(const DTChamberId &dtCh, string histoTag) {

      // n max channels
      int nWires_max = (digiHistos[histoTag])[dtCh.rawId()] -> getNbinsX();

      // set bin content = -1 for each not real channel. For visualization purposes
      for(int sl=1; sl<=3; sl++) {
        for(int ly=1; ly<=4; ly++) {
          for(int ch=1; ch<=nWires_max; ch++) {

            int dduId = -1, rosId = -1, robId = -1, tdcId = -1, channelId = -1;
            int realCh = mapping->geometryToReadOut(dtCh.wheel(),dtCh.station(),dtCh.sector(),sl,ly,ch,dduId,rosId,robId,tdcId,channelId);

            // realCh = 0 if the channel exists, while realCh = 1 if it does not exist
            if( realCh ) {

              int lybin = (4*sl - 4) + ly;
              (digiHistos[histoTag])[dtCh.rawId()] -> setBinContent(ch,lybin,-1.);

            } 

          }
        }
      }

}

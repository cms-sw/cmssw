/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2006/10/08 16:00:24 $
 * $Revision: 1.12 $
 * \author M. Zanetti - INFN Padova
 *
 */

#include <DQM/DTMonitorModule/interface/DTDigiTask.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

// Digis
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

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

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

DTDigiTask::DTDigiTask(const edm::ParameterSet& ps){

  cout<<"[DTDigiTask]: Constructor"<<endl;

  outputFile = ps.getUntrackedParameter<string>("outputFile", "DTDigiSources.root");

  logFile.open("DTDigiTask.log");

  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  edm::Service<MonitorDaemon> daemon; 	 
  daemon.operator->();

  dbe->setVerbose(1);


}


DTDigiTask::~DTDigiTask(){

  cout << "DTDigiTask: analyzed " << nevents << " events" << endl;

  logFile.close();
  if ( (outputFile.size() != 0) && (parameters.getUntrackedParameter<bool>("writeHisto", true)) ) dbe->save(outputFile);
}


void DTDigiTask::beginJob(const edm::EventSetup& context){

  cout<<"[DTDigiTask]: BeginJob"<<endl;

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // Get the pedestals 
  // tTrig 
  if (parameters.getUntrackedParameter<bool>("readDB", true)) 
    context.get<DTTtrigRcd>().get(tTrigMap);
  // t0s 
  if (parameters.getParameter<bool>("performPerWireT0Calibration")) 
    context.get<DTT0Rcd>().get(t0Map);

  // tMax (not yet from the DB)
  tMax = parameters.getParameter<int>("defaultTmax");
  if(parameters.getUntrackedParameter<bool>("MTCC", false))
    {
      for(int wheel=1; wheel<3; wheel++)
	{
	  for(int sec=10; sec<10+wheel; sec++)
	    {
	      for (int st=1; st<5; st++)
		{
		  DTChamberId chId(wheel, st, sec);
		  if ( parameters.getUntrackedParameter<bool>("localrun", true) )
		    {
		      if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) 
			bookHistos( chId, string("TimeBoxes"),"TimeBoxAllHits_perCh");  
		      else
			bookHistos( chId, string("TimeBoxes"),"TimeBoxInTimeHits_perCh" );
		    }
		  if (!parameters.getUntrackedParameter<bool>("localrun", true) )
		    {
		      if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
			bookHistos( chId, string("TimeBoxes"),"TimeBoxAllHits_perChNoDT");  
			bookHistos( chId, string("TimeBoxes"),"TimeBoxAllHits_perChDTalso");  
			bookHistos( chId, string("TimeBoxes"),"TimeBoxAllHits_perChDTonly");  
		      }
		      else {
			bookHistos( chId, string("TimeBoxes"),"TimeBoxInTimeHits_perChNoDT" );
			bookHistos( chId, string("TimeBoxes"),"TimeBoxInTimeHits_perChDTalso" );
			bookHistos( chId, string("TimeBoxes"),"TimeBoxInTimeHits_perChDTonly" );
		      }
		    }
		  if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) 
		    bookHistos( chId, string("Occupancies"),"OccupancyAllHits_perCh"  );
		  else 
		    bookHistos( chId, string("Occupancies"),"OccupancyInTimeHits_perCh" );

		  for(int sl=1; sl<4; sl++)
		    {
		      if(st==4 && sl==2)
			continue;
		      DTSuperLayerId slId(chId, sl);
		      if (parameters.getUntrackedParameter<bool>("localrun", true) )
		      {
			if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) 
			  bookHistos( slId, string("TimeBoxes"), "TimeBoxAllHits" );
			else
			  bookHistos( slId, string("TimeBoxes"), "TimeBoxInTimeHits" );
		      }
		      if (!parameters.getUntrackedParameter<bool>("localrun", true) ) {
			if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
			  bookHistos( slId, string("TimeBoxes"), "TimeBoxAllHitsNoDT" );
			  bookHistos( slId, string("TimeBoxes"), "TimeBoxAllHitsDTalso" );
			  bookHistos( slId, string("TimeBoxes"), "TimeBoxAllHitsDTonly" );
			}
			else{
			  bookHistos( slId, string("TimeBoxes"), "TimeBoxInTimeHitsNoDT" );
			  bookHistos( slId, string("TimeBoxes"), "TimeBoxInTimeHitsDTalso" );
			  bookHistos( slId, string("TimeBoxes"), "TimeBoxInTimeHitsDTonly" );
			}
		      }
		      bookHistos( slId, string("CathodPhotoPeaks"), "CathodPhotoPeak" );
		      for(int layer=1; layer<5; layer++)
			{
			  DTLayerId layerId(slId, layer);
			   if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) 
			     bookHistos( layerId, "OccupancyAllHits" );
			   else
			     bookHistos( layerId, "OccupancyInTimeHits" );
			   bookHistos( layerId, "OccupancyNoise" ); 
			   bookHistos( layerId, "OccupancyAfterPulseHits" );
			}
		    }
		}
	    }

	  DTChamberId chId(wheel, 4, 14);
	  if (parameters.getUntrackedParameter<bool>("localrun", true))
	    {
	      	if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) 
		  bookHistos( chId, string("TimeBoxes"),"TimeBoxAllHits_perCh");  
		else
		  bookHistos( chId, string("TimeBoxes"),"TimeBoxInTimeHits_perCh" );
	    }
	  if (!parameters.getUntrackedParameter<bool>("localrun", true) )
	    {
	      if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)){ 
		bookHistos( chId, string("TimeBoxes"),"TimeBoxAllHits_perChNoDT");  
		bookHistos( chId, string("TimeBoxes"),"TimeBoxAllHits_perChDTalso");  
		bookHistos( chId, string("TimeBoxes"),"TimeBoxAllHits_perChDTonly");  
	      }
	      else {
		bookHistos( chId, string("TimeBoxes"),"TimeBoxInTimeHits_perChNoDT" );
		bookHistos( chId, string("TimeBoxes"),"TimeBoxInTimeHits_perChDTalso" );
		bookHistos( chId, string("TimeBoxes"),"TimeBoxInTimeHits_perChDTonly" );
	      }
	    }
	  if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) 
	    bookHistos( chId, string("Occupancies"),"OccupancyAllHits_perCh"  );
	  else
	    bookHistos( chId, string("Occupancies"),"OccupancyInTimeHits_perCh" );

	  for(int sl=1; sl<4; sl++)
	    {
	      if(sl==2)
		continue;
	      DTSuperLayerId slId(chId, sl);
	      if (parameters.getUntrackedParameter<bool>("localrun", true) )
	      {
		if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) 
		  bookHistos( slId, string("TimeBoxes"), "TimeBoxAllHits" );
		else
		  bookHistos( slId, string("TimeBoxes"), "TimeBoxInTimeHits" );
	      }
	      if (!parameters.getUntrackedParameter<bool>("localrun", true) ) {
		if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
		  bookHistos( slId, string("TimeBoxes"), "TimeBoxAllHitsNoDT" );
		  bookHistos( slId, string("TimeBoxes"), "TimeBoxAllHitsDTalso" );
		  bookHistos( slId, string("TimeBoxes"), "TimeBoxAllHitsDTonly" );
		}
		else {
		  bookHistos( slId, string("TimeBoxes"), "TimeBoxInTimeHitsNoDT" );
		  bookHistos( slId, string("TimeBoxes"), "TimeBoxInTimeHitsDTalso" );
		  bookHistos( slId, string("TimeBoxes"), "TimeBoxInTimeHitsDTonly" );
		}
	      }
	      bookHistos( slId, string("CathodPhotoPeaks"), "CathodPhotoPeak" );
	      for(int layer=1; layer<5; layer++)
		{
		  DTLayerId layerId(slId, layer);
		  if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) 
		    bookHistos( layerId, "OccupancyAllHits" );
		  else
		    bookHistos( layerId, "OccupancyInTimeHits" );
		  bookHistos( layerId, "OccupancyNoise" ); 
		  bookHistos( layerId, "OccupancyAfterPulseHits" );
		}
	    }

	}
    }

}


void DTDigiTask::bookHistos(const DTLayerId& dtLayer, string histoTag) {

  stringstream wheel; wheel << dtLayer.wheel();	
  stringstream station; station << dtLayer.station();	
  stringstream sector; sector << dtLayer.sector();	
  stringstream superLayer; superLayer << dtLayer.superlayer();	
  stringstream layer; layer << dtLayer.layer();	

  cout<<"[DTDigiTask]: booking"<<endl;

  dbe->setCurrentFolder("DT/DTDigiTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/Occupancies");

  cout<<"[DTDigiTask]: histoTag "<<histoTag<<endl;

  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str() 
    + "_L" + layer.str();

  cout<<"[DTDigiTask]: histoName "<<histoName<<endl;

  const int nWires = muonGeom->layer(dtLayer)->specificTopology().channels();

  (digiHistos[histoTag])[dtLayer.rawId()] = dbe->book1D(histoName,histoName,nWires,1,nWires+1);

}




void DTDigiTask::bookHistos(const DTSuperLayerId& dtSL, string folder, string histoTag) {


  stringstream wheel; wheel << dtSL.wheel();	
  stringstream station; station << dtSL.station();	
  stringstream sector; sector << dtSL.sector();	
  stringstream superLayer; superLayer << dtSL.superlayer();	

  cout<<"[DTDigiTask]: booking"<<endl;

  dbe->setCurrentFolder("DT/DTDigiTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/" + folder);

  cout<<"[DTDigiTask]: folder "<< "DT/DTDigiTask/Wheel" + wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/" + folder<<endl;

  cout<<"[DTDigiTask]: histoTag "<<histoTag<<endl;

  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str(); 


  cout<<"[DTDigiTask]: histoName "<<histoName<<endl;

  if ( parameters.getUntrackedParameter<bool>("readDB", false) ) 
    tTrigMap->slTtrig( dtSL, tTrig, tTrigRMS); 
  else tTrig = parameters.getParameter<int>("defaultTtrig");
  

  if ( folder == "TimeBoxes") {
    string histoTitle = histoName + " (TDC Counts)";
    if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
      int maxTDCCounts = 6400 * parameters.getUntrackedParameter<int>("tdcRescale", 1);

      (digiHistos[histoTag])[dtSL.rawId()] = 
	dbe->book1D(histoName,histoTitle, 
		    maxTDCCounts/parameters.getUntrackedParameter<int>("timeBoxGranularity",4), 0, maxTDCCounts);
      
    }    
    else {
      (digiHistos[histoTag])[dtSL.rawId()] = 
	dbe->book1D(histoName,histoTitle, 
		    2*tMax/parameters.getUntrackedParameter<int>("timeBoxGranularity",4), tTrig-tMax, tTrig+2*tMax);
    }
  }

  if ( folder == "CathodPhotoPeaks" && !parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
    (digiHistos[histoTag])[dtSL.rawId()] = dbe->book1D(histoName,histoName,500,0,1000);
  }

  /// FIXME: patch to provide tTrig to the Client. TO BE REMOVED once the ES will be accesible
  if ( folder == "tTrigRef" && !parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
    (digiHistos[histoTag])[dtSL.rawId()] = dbe->book1D(histoName,histoName,10000,0,10000);
  }

}

void DTDigiTask::bookHistos(const DTChamberId& dtCh, string folder, string histoTag) {


  stringstream wheel; wheel << dtCh.wheel();	
  stringstream station; station << dtCh.station();	
  stringstream sector; sector << dtCh.sector();	

  cout<<"[DTDigiTask]: booking"<<endl;

  dbe->setCurrentFolder("DT/DTDigiTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/" + folder);

  cout<<"[DTDigiTask]: folder "<< "DT/DTDigiTask/Wheel" + wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/" + folder<<endl;

  cout<<"[DTDigiTask]: histoTag "<<histoTag<<endl;

  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 

  cout<<"[DTDigiTask]: histoName "<<histoName<<endl;

  if ( folder == "TimeBoxes") {
    DTSuperLayerId slId(dtCh,1);
    if ( parameters.getUntrackedParameter<bool>("readDB", false) ) 
      tTrigMap->slTtrig( slId, tTrig, tTrigRMS); 
    else tTrig = parameters.getParameter<int>("defaultTtrig");
 
    string histoTitle = histoName + " (TDC Counts)";
    if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
      int maxTDCCounts = 6400 * parameters.getUntrackedParameter<int>("tdcRescale", 1);
      
      (digiHistos[histoTag])[dtCh.rawId()] = 
	dbe->book1D(histoName,histoTitle, 
		    maxTDCCounts/parameters.getUntrackedParameter<int>("timeBoxGranularity",4), 0, maxTDCCounts);     
    }    
    else {
      (digiHistos[histoTag])[dtCh.rawId()] = 
	dbe->book1D(histoName,histoTitle, 
		    2*tMax/parameters.getUntrackedParameter<int>("timeBoxGranularity",4), tTrig-tMax, tTrig+2*tMax);
    }
  }
  
   else if ( folder == "Occupancies")
   {
     int station = dtCh.station();
          int nWires=0;
     if(station==1 || station==2 || station==4)
       nWires = 62;
     else if(station==3)
       nWires = 72;
     (digiHistos[histoTag])[dtCh.rawId()] = dbe->book2D(histoName,histoName,nWires,1,nWires+1,12,0,12);
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


void DTDigiTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;
  if (nevents%1000 == 0) 
    cout<<"[DTDigiTask]: "<<nevents<<" events analyzed"<<endl;

  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel("dtunpacker", dtdigis);

  if ( !parameters.getUntrackedParameter<bool>("localrun", true) ) e.getByType(ltcdigis);
  
  bool checkNoisyChannels = parameters.getUntrackedParameter<bool>("checkNoisyChannels","false");
  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    // Get the map of noisy channels
    c.get<DTStatusFlagRcd>().get(statusMap);
  }

  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
   for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){
 
   if(checkNoisyChannels) {
	const DTWireId wireId(((*dtLayerId_It).first), (*digiIt).wire());
	bool isNoisy = false;
	bool isFEMasked = false;
	bool isTDCMasked = false;
	bool isTrigMask = false;
	bool isDead = false;
	bool isNohv = false;
	statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
	if(isNoisy) {
	  continue;
	}      
      }

    // for clearness..
      uint32_t indexL = ((*dtLayerId_It).first).rawId();
      int layer_number=((*dtLayerId_It).first).layer();
      const  DTSuperLayerId dtSLId = ((*dtLayerId_It).first).superlayerId();
      uint32_t indexSL = dtSLId.rawId();
      int superlayer_number=dtSLId.superlayer();

      const  DTChamberId dtChId = dtSLId.chamberId(); 
      uint32_t indexCh = dtChId.rawId();
     
      if ( parameters.getUntrackedParameter<bool>("readDB", false) ) 
	tTrigMap->slTtrig( ((*dtLayerId_It).first).superlayerId(), tTrig, tTrigRMS); 
      else tTrig = parameters.getParameter<int>("defaultTtrig");
      
      int inTimeHitsLowerBound = int(round(tTrig)) - parameters.getParameter<int>("inTimeHitsLowerBound");
      int inTimeHitsUpperBound = int(round(tTrig)) + tMax + parameters.getParameter<int>("inTimeHitsUpperBound");

      float t0; float t0RMS;
      int tdcTime = (*digiIt).countsTDC();

      if (parameters.getParameter<bool>("performPerWireT0Calibration")) {
	const DTWireId dtWireId(((*dtLayerId_It).first), (*digiIt).wire());
	t0Map->cellT0(dtWireId, t0, t0RMS) ;
	tdcTime += int(round(t0));
      }


      /* * * * * * * * * * * * * * */
      /* S T A R T   F I L L I N G */
      /* * * * * * * * * * * * * * */

      string histoTag;
 
      // only for pre-Calibration jobs 
      if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
	// Occupancies
	histoTag = "OccupancyAllHits";
	if ((digiHistos[histoTag].find(indexL) == digiHistos[histoTag].end()) && 
	    !parameters.getUntrackedParameter<bool>("MTCC", false)) 
	  bookHistos( (*dtLayerId_It).first, histoTag);

	(digiHistos.find(histoTag)->second).find(indexL)->second->Fill((*digiIt).wire());

	//Occupancies per chamber
	histoTag = "OccupancyAllHits_perCh";
	if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()) && 
	    !parameters.getUntrackedParameter<bool>("MTCC", false)) 
	  bookHistos( dtChId, string("Occupancies"), histoTag );
	
	(digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*digiIt).wire(),(layer_number+(superlayer_number-1)*4)-1);

	// TimeBoxes
	histoTag = "TimeBoxAllHits" + triggerSource();
	if ((digiHistos[histoTag].find(indexSL) == digiHistos[histoTag].end()) && 
	    !parameters.getUntrackedParameter<bool>("MTCC", false)  )
	  bookHistos( dtSLId, string("TimeBoxes"), histoTag );

	(digiHistos.find(histoTag)->second).find(indexSL)->second->Fill(tdcTime);
	
	//Time Boxes per chamber
	histoTag = "TimeBoxAllHits_perCh" + triggerSource();
	if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end())  && 
	    !parameters.getUntrackedParameter<bool>("MTCC", false)  )
	  bookHistos( dtChId, string("TimeBoxes"), histoTag );
	
	(digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(tdcTime);
      }

      // after-Calibration jobs 
      else {

	// Noise: Before tTrig
	if (tdcTime < inTimeHitsLowerBound ) {
	  histoTag = "OccupancyNoise";
	  if ((digiHistos[histoTag].find(indexL) == digiHistos[histoTag].end())  && 
	    !parameters.getUntrackedParameter<bool>("MTCC", false)  )
	    bookHistos((*dtLayerId_It).first, histoTag); 
	  
	  (digiHistos.find(histoTag)->second).find(indexL)->second->Fill((*digiIt).wire());
	  }
	
	// time > tTrig
	else if (tdcTime > inTimeHitsLowerBound ) { 
	
	  // Physical hits: into the time window	
	  if ( tdcTime < inTimeHitsUpperBound) {
	  
	    // Occupancies
	    histoTag = "OccupancyInTimeHits";
	    if ((digiHistos[histoTag].find(indexL) == digiHistos[histoTag].end()) && 
	    !parameters.getUntrackedParameter<bool>("MTCC", false)  )
	      bookHistos( (*dtLayerId_It).first, histoTag );

	    (digiHistos.find(histoTag)->second).find(indexL)->second->Fill((*digiIt).wire());
	  
	    //Occupancies per chamber
	    histoTag = "OccupancyInTimeHits_perCh";
	    if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end())  && 
		!parameters.getUntrackedParameter<bool>("MTCC", false)  ){
	      bookHistos( dtChId, string("Occupancies"), histoTag );
	    }
	    (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*digiIt).wire(),(layer_number+(superlayer_number-1)*4)-1);
	    
	    // TimeBoxes
	    histoTag = "TimeBoxInTimeHits" + triggerSource();
	    if ((digiHistos[histoTag].find(indexSL) == digiHistos[histoTag].end()) && 
		!parameters.getUntrackedParameter<bool>("MTCC", false)  )
	      bookHistos( dtSLId, string("TimeBoxes"), histoTag );

	    (digiHistos.find(histoTag)->second).find(indexSL)->second->Fill(tdcTime);


	    //Time Boxes per chamber
	    histoTag = "TimeBoxInTimeHits_perCh" + triggerSource();
	    if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()) &&
		!parameters.getUntrackedParameter<bool>("MTCC", false)  )
	      bookHistos( dtChId, string("TimeBoxes"), histoTag );

	    (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(tdcTime);

	  
	    // After pulses from a physical hit
	    if ( (*digiIt).number() > 0 ) {
	      histoTag = "OccupancyAfterPulseHits";
	      if ((digiHistos[histoTag].find(indexL) == digiHistos[histoTag].end())  &&
		  !parameters.getUntrackedParameter<bool>("MTCC", false)  )
		bookHistos( (*dtLayerId_It).first, histoTag );

	      (digiHistos.find(histoTag)->second).find(indexL)->second->Fill((*digiIt).wire());
	    }
	  
	  }
	
	  // After pulses: after the time window
	  if ( tdcTime > inTimeHitsUpperBound ) {
	    histoTag = "OccupancyAfterPulseHits";
	    if ((digiHistos[histoTag].find(indexL) == digiHistos[histoTag].end())  &&
		!parameters.getUntrackedParameter<bool>("MTCC", false)  )
	      bookHistos( (*dtLayerId_It).first, histoTag );

	    (digiHistos.find(histoTag)->second).find(indexL)->second->Fill((*digiIt).wire());
	  }
	
	}

	// 2nd - 1st
	if ( (*digiIt).number() == 1 ) {
	 
	  DTDigiCollection::const_iterator firstDigiIt = digiIt;
	  firstDigiIt--;

	  histoTag = "CathodPhotoPeak";
	  if ((digiHistos[histoTag].find(indexSL) == digiHistos[histoTag].end())  &&
	      !parameters.getUntrackedParameter<bool>("MTCC", false)  )
	    bookHistos( dtSLId, string("CathodPhotoPeaks"), histoTag );

	  (digiHistos.find(histoTag)->second).find(indexSL)->second->Fill((*digiIt).countsTDC()-
									  (*firstDigiIt).countsTDC());
	}
       

	// Fake tTrig histo
	if (digiHistos[string("tTrigRef")].find(indexSL) != 
	    digiHistos[string("tTrigRef")].end()) {
	  (digiHistos.find(string("tTrigRef"))->second).find(indexSL)->second->Fill(tTrig);
	} else {
	  bookHistos( dtSLId, string("tTrigRef"), string("tTrigRef") );
	  (digiHistos.find(string("tTrigRef"))->second).find(indexSL)->second->Fill(tTrig);
	} 

      }
    }
  }
  
}


string DTDigiTask::triggerSource() {

  string l1ASource;

  if ( !parameters.getUntrackedParameter<bool>("localrun", true) ){
    
    for (std::vector<LTCDigi>::const_iterator ltc_it = ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
      
      int otherTriggerSum=0;
      for (int i = 1; i < 6; i++)
	otherTriggerSum += int((*ltc_it).HasTriggered(i));
      
      if ((*ltc_it).HasTriggered(0) && otherTriggerSum == 0) 
	l1ASource = "DTonly";
      else if (!(*ltc_it).HasTriggered(0))
	l1ASource = "NoDT";
      //      else 
      //	l1ASource = "DTalso";
      else if ((*ltc_it).HasTriggered(0) && otherTriggerSum > 0)
	l1ASource = "DTalso";
    }
  }
  return l1ASource;
}


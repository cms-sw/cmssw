/*
 * \file DTLocalTriggerTask.cc
 * 
 * $Date: 2007/04/12 14:05:12 $
 * $Revision: 1.4 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "DQM/DTMonitorModule/interface/DTLocalTriggerTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Digis
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"

// DT trigger
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"



//Digis & RecHit
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>


using namespace edm;
using namespace std;

DTLocalTriggerTask::DTLocalTriggerTask(const edm::ParameterSet& ps){
  
  debug = ps.getUntrackedParameter<bool>("debug", "false");
  if(debug)   cout<<"[DT/DTLocalTriggerTask]: Constructor"<<endl;

  outputFile = ps.getUntrackedParameter<string>("outputFile", "DTLocalTriggerSources.root");
  logFile.open("DTLocalTriggerTask.log");

  dcc_label = ps.getUntrackedParameter<string>("dcc_label", "dttpgprod");
  ros_label = ps.getUntrackedParameter<string>("ros_label", "dtunpacker");
  seg_label = ps.getUntrackedParameter<string>("seg_label", "dt4DSegments");
  
  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon; 	 
  daemon.operator->();    
  if ( dbe ) {
    dbe->setVerbose(1);
    dbe->setCurrentFolder("DT/DTLocalTriggerTask");
    runId = dbe->bookInt("iRun");
  }

}


DTLocalTriggerTask::~DTLocalTriggerTask() {

if(debug)
  cout << "DTLocalTriggerTask: analyzed " << nevents << " events" << endl;
 logFile.close();

}

void DTLocalTriggerTask::beginJob(const edm::EventSetup& c){

 if(debug)
    cout<<"[DTLocalTriggerTask]: BeginJob"<<endl;

  nevents = 0;
  
}

void DTLocalTriggerTask::endJob(){

  cout << "DTLocalTriggerTask: analyzed " << nevents << " events" << endl;
  if ( (outputFile.size() != 0) && (parameters.getUntrackedParameter<bool>("writeHisto", true)) ) 
    dbe->save(outputFile);
  dbe->rmdir("DT/DTLocalTriggerTask");

}

void DTLocalTriggerTask::analyze(const edm::Event& e, const edm::EventSetup& c){
  
  nevents++;
  string histoType ;
  string histoTag ;

  int phcode_best[6][5][13];  
  vector<L1MuDTChambPhDigi>::const_iterator ibest[6][5][13];

  int dduphcode_best[6][5][13];  
  //vector<DTLocalTriggerCollection>::const_iterator dduibest[6][5][13]; 
    
  if ( !parameters.getUntrackedParameter<bool>("localrun", true) ) e.getByType(ltcdigis);
  string trigsrc= triggerSource();
  

  if (parameters.getUntrackedParameter<bool>("process_dcc", true) ) {
    
    ///////////////////////////////
    /* SM LOCAL TRIGGER PHI VIEW */
    ///////////////////////////////
    
    edm::Handle<L1MuDTChambPhContainer> l1dtlocalphi;
    e.getByLabel(dcc_label, l1dtlocalphi);
    vector<L1MuDTChambPhDigi>*  l1phitrig = l1dtlocalphi->getContainer();
    // define best quality phi trigger segment in any station
 // start from 1 and zero is kept empty
    for (int i=0;i<5;++i)
      for (int j=0;j<6;++j)
	for (int k=0;k<13;++k)
	  phcode_best[j][i][k] = -1;

    
    for(vector<L1MuDTChambPhDigi>::const_iterator i = l1phitrig->begin(); i != l1phitrig->end(); i++) {
      int phwheel = i->whNum();
      int phsec   = i->scNum() + 1; // SM The track finder goes from 0 to 11. I need them from 1 to 12 !!!!!
      int phst    = i->stNum();
      int phbx    = i->bxNum();
      int phcode  = i->code();
      int phi1st  = i->Ts2Tag();
      int phphi   = i->phi();
      int phphiB  = i->phiB();
      
      if(phcode>phcode_best[phwheel+3][phst][phsec] && phcode<7) {
	phcode_best[phwheel+3][phst][phsec]=phcode; 
	ibest[phwheel+3][phst][phsec] = i;
      }
      
      DTChamberId dtChId(phwheel,phst,phsec);  // get chamber for histograms
      uint32_t indexCh = dtChId.rawId();
      
      // SM BX vs Quality Phi view
      histoType = "BXvsQual" ;
      histoTag = histoType + trigsrc;

      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phcode,phbx);
      
      // SM 1st2ndflag vs Quality Phi view
      histoType = "TS2TagCodevsQual" ;
      histoTag = histoType + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phcode,phi1st);
      
      // SM BX vs Quality Phi view
      histoTag = histoType + trigsrc;
      histoType = "QualvsPhirad" ;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phphi,phcode);
      
      // SM BX vs Quality Phi view
      histoType = "QualvsPhibend" ;
      histoTag = histoType + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phphiB,phcode);
      
      if(phi1st==0) {
	// SM BX 1st trigger track segment, phi view
	histoType = "BX1stSegment" ;
	histoTag = histoType + trigsrc;
	if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	  bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
	(digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phbx);
	
	// SM Quality 1st trigger track segment, phi view
	histoType = "Qual1stSegment" ;
	histoTag = histoType + trigsrc;
	if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	  bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
	(digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phcode);
	
      }
      else if(phi1st==1) {
	// SM BX 2nd trigger track segment, phi view
	histoType = "BX2ndSegment" ;
	histoTag = histoType + trigsrc;
	if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	  bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
	(digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phbx);
	
	// SM Quality 2nd trigger track segment, phi view
	histoType = "Qual2ndSegment" ;
	histoTag = histoType + trigsrc;
	if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	  bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
	(digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phcode);
      }
    } 
    
    /////////////////////////////////
    /* SM LOCAL TRIGGER THETA VIEW */ 
    /////////////////////////////////
    
    edm::Handle<L1MuDTChambThContainer> l1dtlocalth;
    e.getByLabel(dcc_label, l1dtlocalth);
    vector<L1MuDTChambThDigi>*  l1thetatrig = l1dtlocalth->getContainer();
    int thcode[7];

    for(vector<L1MuDTChambThDigi>::const_iterator j = l1thetatrig->begin(); j != l1thetatrig->end(); j++) {
      int thwheel = j->whNum();
      int thsec   = j->scNum() + 1; // SM The track finder goes from 0 to 11. I need them from 1 to 12 !!!!!
      int thst    = j->stNum();
      int thbx    = j->bxNum();
      
      for (int pos=0; pos<7; pos++) {
	thcode[pos]  = j->code(pos);
      } 
      
      DTChamberId dtChId(thwheel,thst,thsec);  // get chamber for histograms
      uint32_t indexCh = dtChId.rawId();   
      
      // SM BX vs Position Theta view
      histoType = "PositionvsBX" ;
      histoTag = histoType + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerTheta"), histoType, trigsrc );
      for (int pos=0; pos<7; pos++) //SM fill position for non zero position bit in theta view
	if(thcode[pos]>0)
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(thbx,pos);
      
      // SM BX vs Position Theta view
      histoType = "PositionvsCode" ;
      histoTag = histoType + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerTheta"), histoType, trigsrc );
      for (int pos=0; pos<7; pos++) //SM fill position for non zero position bit in theta view
	if(thcode[pos]>0)
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(thcode[pos],pos);
      
      // SM BX vs Code Theta view
      histoType = "CodevsBX" ;
      histoTag = histoType + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerTheta"), histoType, trigsrc );
      for (int pos=0; pos<7; pos++) //SM fill position for non zero position bit in theta view
	if(thcode[pos]>0) 
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(thbx,thcode[pos]);
    }
  }  


  if ( parameters.getUntrackedParameter<bool>("process_ros", true) ) {
    ////////////////////////////////////////////////////////////////////
    /* SM DT Local Trigger as in MTCC (Francesca Cavallo's unpacking) */  
    ////////////////////////////////////////////////////////////////////
    
    Handle<DTLocalTriggerCollection> dtTrigs;
    e.getByLabel(ros_label,dtTrigs);
    
    DTLocalTriggerCollection::DigiRangeIterator detUnitIt;
    
    // define best quality ddu phi trigger segment in any station
 // start from 1 and zero is kept empty
    for (int i=0;i<5;++i)
      for (int j=0;j<6;++j)
	for (int k=0;k<13;++k)
	  dduphcode_best[j][i][k] = -1;


    for (detUnitIt=dtTrigs->begin();
	 detUnitIt!=dtTrigs->end();
	 ++detUnitIt){
      
      const DTChamberId& id = (*detUnitIt).first;
      const DTLocalTriggerCollection::Range& range = (*detUnitIt).second;
      uint32_t indexCh = id.rawId();  
      
      // Loop over the trigger segments  
      
      for (DTLocalTriggerCollection::const_iterator trigIt = range.first;
	   trigIt!=range.second;
	   ++trigIt){
	
	int bx = trigIt->bx();
	int quality = trigIt->quality();
        int thqual = trigIt->trTheta();
        int flag1st = 0;
	    
	int wh = id.wheel();
	int sec = id.sector();
	int st = id.station();

	
	//	if(quality<7 && thqual==0 ) {	  // it is a phi trigger
	if( quality<7 ) {	  // it is a phi trigger

	  if(quality>dduphcode_best[wh+3][st][sec] && quality<7) { // find best ddu trigger in phi view
	    dduphcode_best[wh+3][st][sec]=quality; 
//	    dduibest[wh+3][st][sec] = trigIt;    // SM commentato
	  }
	  
	  histoType = "DDU_BXvsQual" ;
	  histoTag = histoType + trigsrc;
	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	    bookHistos( id, string("LocalTriggerPhi"), histoType, trigsrc );
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(quality,bx);

          if(trigIt->secondTrack())  flag1st = 1;  // it is a second trigger track
           
	  histoType = "DDU_Flag1stvsBX" ;
	  histoTag = histoType + trigsrc;
	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	    bookHistos( id, string("LocalTriggerPhi"), histoType, trigsrc );
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(bx,flag1st);

	  histoType = "DDU_Flag1stvsQual";
	  histoTag = histoType + trigsrc;
	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	    bookHistos( id, string("LocalTriggerPhi"), histoType, trigsrc );
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(quality,flag1st);	  
	}

	//	else if(quality==7 && thqual>0) {  // it is a theta trigger
	else if( thqual>0 ) {  // it is a theta trigger

 	  string histoType = "DDU_BXvsThQual" ;
	  histoTag = histoType + trigsrc;
 	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
 	    bookHistos( id, string("LocalTriggerTheta"), histoType, trigsrc );
 	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(thqual,bx);
 	}
      }   
    } 
  }
 
  if ( parameters.getUntrackedParameter<bool>("process_seg", true) ) {
    
    ///////////////////////////////////////////////////////////
    /* SM Comparison with reconstructed local track segments */
    ///////////////////////////////////////////////////////////
    
    // Get the 4D segment collection from the event
    Handle<DTRecSegment4DCollection> all4DSegments;
    e.getByLabel(seg_label, all4DSegments);  
    DTRecSegment4DCollection::const_iterator track;
    // it tells whether there is a track in a station.
    Bool_t track_flag[6][5][15]; 
    memset(track_flag,false,450*sizeof(bool));
    
    for ( track = all4DSegments->begin(); track != all4DSegments->end(); ++track){
      
      if((*track).hasPhi()) { // Phi component
	
	int wheel = (*track).chamberId().wheel();
	int sector = (*track).chamberId().sector();
	int station = (*track).chamberId().station();
	int scsector;
	switch (sector) {
	case 13:
	  scsector = 4;
	  break;
	case 14:
	  scsector = 10;
	  break;
	default:
	  scsector = sector;
	}
	
	
	if(!track_flag[wheel+3][station][sector]) { // if no track already found in this station
	  track_flag[wheel+3][station][sector] = true;  // the 1st track is always the best
	  
	  DTChamberId dtChId(wheel,station,scsector);  // get chamber for histograms
	  uint32_t indexCh = dtChId.rawId();   
	  
	  // position of track for reconstruced tracks (denom. for trigger efficiency)
	  histoType = "Track_pos" ;
	  histoTag = histoType + trigsrc;
	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	    bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*track).localPosition().x());
	  
	  if (parameters.getUntrackedParameter<bool>("process_dcc", true) ) {	    
	    if (phcode_best[wheel+3][station][scsector] > -1 && phcode_best[wheel+3][station][scsector] < 7) {
	      
	      histoType = "Track_pos_andtrig" ;
	      histoTag = histoType + trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*track).localPosition().x());
	      
	      // SM phi of the track vs phi of the trigger
	      histoType = "PhitkvsPhitrig" ;
	      histoTag = histoType + trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*ibest[wheel+3][station][scsector]).phi(),(*track).localPosition().x());
	      
	      // SM hits of the track vs quality of the trigger
	      histoType = "HitstkvsQualtrig" ;
	      histoTag = histoType + trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*ibest[wheel+3][station][scsector]).code(),(*track).degreesOfFreedom() );
	    }
	  }
	  if ( parameters.getUntrackedParameter<bool>("process_ros", true) ) {	    
	    if (dduphcode_best[wheel+3][station][scsector] > -1 && dduphcode_best[wheel+3][station][scsector] < 7) {
	      
	      histoType = "DDU_Track_pos_andtrig" ;
	      histoTag = histoType + trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("LocalTriggerPhi"), histoType, trigsrc );
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*track).localPosition().x());
	    }
	  }
	}
      }
    }  
  }
  

  
}

void DTLocalTriggerTask::bookHistos(const DTChamberId& dtCh, string folder, string histoType, string trigSource) {


  stringstream wheel; wheel << dtCh.wheel();	
  stringstream station; station << dtCh.station();	
  stringstream sector; sector << dtCh.sector();	
  if (debug)
    cout<<"[DTLocalTriggerTask]: booking"<<endl;

  dbe->setCurrentFolder("DT/DTLocalTriggerTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/" + folder);
  
  if (debug){
    cout<<"[DTLocalTriggerTask]: folder "<< "DT/DTLocalTriggerTask/Wheel" + wheel.str() +
      "/Station" + station.str() +
      "/Sector" + sector.str() + "/" + folder<<endl;
    cout<<"[DTLocalTriggerTask]: histoType "<<histoType<<endl;
  }
  
  string histoTag = histoType + trigSource;
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 
  cout<<"     histoName "<<histoName<< " ---- histotag in bookhisto = " << histoTag << endl;
  
  if (debug) 
    cout<<"[DTLocalTriggerTask]: histoName "<<histoName<<endl;
  
  if ( folder == "LocalTriggerPhi") {
    
    if( histoType == "BXvsQual") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,8,-0.5,7.5,101,-50.5,50.5);
    }
     if( histoType == "TS2TagCodevsQual") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,8,-0.5,7.5,2,-0.5,1.5);
    }  
    if( histoType == "QualvsPhirad") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,500,-1500.,1500.,8,-0.5,7.5);
    }
    if( histoType == "QualvsPhibend") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,100,-50.,50.,8,-0.5,7.5);
    }

    if( histoType == "DDU_BXvsQual") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,8,-0.5,7.5,101,-50.5,50.5);
    }
    if( histoType == "DDU_Flag1stvsBX") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,101,-50.5,50.5,2,-0.5,1.5);
    }
    if( histoType == "DDU_Flag1stvsQual") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,8,-0.5,7.5,2,-0.5,1.5);
    }

    if( histoType == "BX1stSegment") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book1D(histoName,histoName,101,-50.5,50.5);
    }
    if( histoType == "BX2ndSegment") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book1D(histoName,histoName,101,-50.5,50.5);
    }
    if( histoType == "Qual1stSegment") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book1D(histoName,histoName,8,-0.5,7.5);
    }
    if( histoType == "Qual2ndSegment") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book1D(histoName,histoName,8,-0.5,7.5);
    }

    //
    if( histoType == "Track_pos") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book1D(histoName,histoName,50,-150.,150.);
    }
    if( histoType == "Track_pos_andtrig") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book1D(histoName,histoName,50,-150.,150.);
    }
    if( histoType == "DDU_Track_pos_andtrig") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book1D(histoName,histoName,50,-150.,150.);
    }

    if( histoType == "PhitkvsPhitrig") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,500,-1500.,1500.,150,-150.,150.);
    }
    if( histoType == "HitstkvsQualtrig") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,8,-0.5,7.5,10,0.5,10.5);
    }
  }
  
   else if ( folder == "LocalTriggerTheta")   {
    if( histoType == "PositionvsBX") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,101,-50.5,50.5,7,-0.5,6.5);
    }
     if( histoType == "PositionvsCode") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,3,-0.5,2.5,6,-0.5,6.5);
    }  
    if( histoType == "CodevsBX") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,101,-50.5,50.5,3,-0.5,2.5);
    }
    if( histoType == "DDU_BXvsThQual") {
      (digiHistos[histoTag])[dtCh.rawId()] = 
 	dbe->book2D(histoName,histoName,8,-0.5,7.5,101,-50.5,50.5);
    }
   }
}
//SM end

//SM 
string DTLocalTriggerTask::triggerSource() {
  
  string l1ASource;
  
  if ( !parameters.getUntrackedParameter<bool>("localrun", true) ){
    
    for (std::vector<LTCDigi>::const_iterator ltc_it = ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
      
      int otherTriggerSum=0;

      for (int i = 1; i < 6; i++) {
	otherTriggerSum += int((*ltc_it).HasTriggered(i));
      }
      if ((*ltc_it).HasTriggered(0) && otherTriggerSum == 0) 
	l1ASource = "_DTonly";
      else if (!(*ltc_it).HasTriggered(0))
	l1ASource = "_NoDT";
      else if ((*ltc_it).HasTriggered(0) && otherTriggerSum > 0)
	l1ASource = "_DTalso";
      
    }
    return l1ASource;
  }
  return "";
}

//SM end

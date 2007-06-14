/*
 * \file DTLocalTriggerTask.cc
 * 
 * $Date: 2006/10/18 18:00:14 $
 * $Revision: 1.2 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "DQM/DTMonitorModule/interface/DTLocalTriggerTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// Digis
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

// DT trigger
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"

//Digis & RecHit
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include <string>
#include <sstream>

using namespace edm;
using namespace std;

DTLocalTriggerTask::DTLocalTriggerTask(const edm::ParameterSet& ps){
  
  debug = ps.getUntrackedParameter<bool>("debug", "false");
  if(debug)   cout<<"[DTLocalTriggerTask]: Constructor"<<endl;

  outputFile = ps.getUntrackedParameter<string>("outputFile", "DTLocalTriggerSources.root");

  dcc_label = ps.getUntrackedParameter<string>("dcc_label", "dttpgprod");
  ros_label = ps.getUntrackedParameter<string>("ros_label", "dtunpacker");
  seg_label = ps.getUntrackedParameter<string>("seg_label", "dt4DSegments");
  
  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon; 	 
  daemon.operator->();    

}


DTLocalTriggerTask::~DTLocalTriggerTask() {

if(debug)
  cout << "DTLocalTriggerTask: analyzed " << nevents << " events" << endl;

}

void DTLocalTriggerTask::beginJob(const edm::EventSetup& context){

 if(debug)
    cout<<"[DTLocalTriggerTask]: BeginJob"<<endl;

 context.get<MuonGeometryRecord>().get(muonGeom);

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
  int dduphcode_best[6][5][13];  
  vector<L1MuDTChambPhDigi>::const_iterator ibest[6][5][13];

  int thcode_best[6][5][13];  
  int dduthcode_best[6][5][13];    
  vector<L1MuDTChambThDigi>::const_iterator ithbest[6][5][13];
 
    
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
      
      DTChamberId dtChId(phwheel,phst,phsec);
      uint32_t indexCh = dtChId.rawId();
      uint32_t indexScWh = 5*(phsec-1) + (phwheel+3) ;    // wheel + sector identifier for specific histograms
      
      // SM BX vs Quality Phi view
      
      string histoTag = "DCC_BXvsQual"+ trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoTag );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phcode,phbx);
      
      // SM 1st2ndflag vs Quality Phi view
      histoTag = "DCC_TS2TagCodevsQual"+ trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoTag );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phcode,phi1st);
      
      // SM Quality vs radial angle Phi view
      histoTag = "DCC_QualvsPhirad" + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoTag );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phphi,phcode);
      
      // SM Quality vs bending Phi view
      histoTag = "DCC_QualvsPhibend" + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoTag );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phphiB,phcode);
      
      // SM BX 1st trigger track segment, phi view
      histoTag = "DCC_Flag1stvsBX" + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoTag );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phbx,phi1st);
      
      // SM Quality 1st trigger track segment, phi view
      histoTag =  "DCC_Flag1stvsQual" + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerPhi"), histoTag );
      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(phcode,phi1st);
      
    } 
    
    /////////////////////////////////
    /* SM LOCAL TRIGGER THETA VIEW */ 
    /////////////////////////////////
    
    edm::Handle<L1MuDTChambThContainer> l1dtlocalth;
    e.getByLabel(dcc_label, l1dtlocalth);
    vector<L1MuDTChambThDigi>*  l1thetatrig = l1dtlocalth->getContainer();
    int thcode[7];

    for (int i=0;i<5;++i)
      for (int j=0;j<6;++j)
	for (int k=0;k<13;++k)
	  thcode_best[j][i][k] = -1;

    for(vector<L1MuDTChambThDigi>::const_iterator j = l1thetatrig->begin(); j != l1thetatrig->end(); j++) {
      int thwheel = j->whNum();
      int thsec   = j->scNum() + 1; // SM The track finder goes from 0 to 11. I need them from 1 to 12 !!!!!
      int thst    = j->stNum();
      int thbx    = j->bxNum();
      
      for (int pos=0; pos<7; pos++) {
	thcode[pos] = j->code(pos);

	if(thcode[pos]>thcode_best[thwheel+3][thst][thsec] ) {
	  thcode_best[thwheel+3][thst][thsec]=thcode[pos]; 
	  ithbest[thwheel+3][thst][thsec] = j;
	}
      } 
      
      DTChamberId dtChId(thwheel,thst,thsec);
      uint32_t indexCh = dtChId.rawId();   
      uint32_t indexScWh = 5*(thsec-1) + (thwheel+3) ;    // wheel + sector identifier for specific histograms     

      // SM BX vs Position Theta view
      histoTag = "DCC_PositionvsBX"+ trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerTheta"), histoTag );
      for (int pos=0; pos<7; pos++) //SM fill position for non zero position bit in theta view
	if(thcode[pos]>0)
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(thbx,pos);
      
      // SM Code vs Position Theta view
      histoTag =  "DCC_PositionvsCode" + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerTheta"), histoTag );
      for (int pos=0; pos<7; pos++) //SM fill position for non zero position bit in theta view
	if(thcode[pos]>0)
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(thcode[pos],pos);
      
      // SM BX vs Code Theta view
      histoTag =  "DCC_ThetaBXvsQual" + trigsrc;
      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	bookHistos( dtChId, string("LocalTriggerTheta"), histoTag );
      for (int pos=0; pos<7; pos++) //SM fill position for non zero position bit in theta view
	if(thcode[pos]>0){
	  int thqual = (thcode[pos]/2)*2+1;
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(thqual,thbx);
	}
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

    for (int i=0;i<5;++i)
      for (int j=0;j<6;++j)
	for (int k=0;k<13;++k)
	  dduthcode_best[j][i][k] = -1;
    
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

	uint32_t indexScWh = 5*(sec-1) + (wh+3) ;    // wheel + sector identifier for specific histograms      

    // check if SC data exist: fill for any trigger

	
	if(quality<7 ) {	  // it is a phi trigger


	  if(quality>dduphcode_best[wh+3][st][sec] && quality<7) { // find best ddu trigger in phi view
	    
	    if(dduphcode_best[wh+3][st][sec]== -1) {  // see if at least a Phi trigger 
 	      histoTag = "DDU_SCdata" + trigsrc;
	      if ((digiHistos[histoTag].find(indexScWh) == digiHistos[histoTag].end())) {
	        bookHistos( id, string("SectorGeneral"), histoTag);
		 }	    
	      (digiHistos.find(histoTag)->second).find(indexScWh)->second->Fill(st);
	     }
	     
	     dduphcode_best[wh+3][st][sec]=quality; 
	  }
	  
	  histoTag =  "DDU_BXvsQual" + trigsrc;
	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	    bookHistos( id, string("LocalTriggerPhi"), histoTag );
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(quality,bx);

          if(trigIt->secondTrack())  flag1st = 1;  // it is a second trigger track
           
	  histoTag =  "DDU_Flag1stvsBX" + trigsrc;
	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	    bookHistos( id, string("LocalTriggerPhi"), histoTag );
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(bx,flag1st);

	  histoTag = "DDU_Flag1stvsQual" + trigsrc;
	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	    bookHistos( id, string("LocalTriggerPhi"), histoTag );
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(quality,flag1st);	  
	}
	if( thqual>0) {  // it is a theta trigger
	  
	  if(thqual>dduthcode_best[wh+3][st][sec] ) { // find best ddu trigger in theta view
	    dduthcode_best[wh+3][st][sec]=thqual; 
	  }

	  histoTag = "DDU_ThetaBXvsQual" + trigsrc;
 	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
 	    bookHistos( id, string("LocalTriggerTheta"), histoTag );
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
	int scsector = 0;
	float x_track = 0;
	float y_track = 0;
	float x_angle = (*track).localDirection().x()/ (*track).localDirection().z();
	float y_angle = (*track).localDirection().y()/ (*track).localDirection().z();

	LocalPoint lpos;
	const DTChamber* chamb;
	const DTChamber* scchamb;

	switch (sector) {
	case 13:
	  scsector = 4;
	  chamb   = muonGeom->chamber(DTChamberId(wheel,station,sector));
	  scchamb = muonGeom->chamber(DTChamberId(wheel,station,scsector));
	  lpos = scchamb->toLocal(chamb->toGlobal((*track).localPosition()));
	  x_track = lpos.x();
	  y_track = lpos.y();
	  break;
	case 14:
	  scsector = 10;
	  chamb   = muonGeom->chamber(DTChamberId(wheel,station,sector));
	  scchamb = muonGeom->chamber(DTChamberId(wheel,station,scsector));
	  lpos = scchamb->toLocal(chamb->toGlobal((*track).localPosition()));
	  x_track = lpos.x();
	  y_track = lpos.y();
	  break;
	default:
	  scsector = sector;
	  x_track = (*track).localPosition().x();
	  y_track = (*track).localPosition().y();
	}

        	
	if(!track_flag[wheel+3][station][sector]) { // if no track already found in this station
	  track_flag[wheel+3][station][sector] = true;  // the 1st track is always the best
	  
	  DTChamberId dtChId(wheel,station,scsector);  // get chamber for histograms
	  uint32_t indexCh = dtChId.rawId(); 
	  
	  // position of track for reconstruced tracks (denom. for trigger efficiency)
	  histoTag = "SEG_TrackPos" + trigsrc;
	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	    bookHistos( dtChId, string("Segment"), histoTag );        
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(x_track);
	  histoTag =  "SEG_TrackAngle"+ trigsrc;
	  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	    bookHistos( dtChId, string("Segment"), histoTag );        
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(x_angle);
	  
	  if((*track).hasZed()) { // Theta component
	    
	    // position of track for reconstruced tracks (denom. for trigger efficiency) along theta direction
	    histoTag = "SEG_TrackThetaPos" + trigsrc;
	    if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	      bookHistos( dtChId, string("Segment"), histoTag );        
	    (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(y_track);
	    
	    histoTag = "SEG_TrackThetaAngle" + trigsrc;
	    if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
	      bookHistos( dtChId, string("Segment"), histoTag );        
	    (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(y_angle);
	  }
	  
	  if (parameters.getUntrackedParameter<bool>("process_dcc", true) ) {
	    if (phcode_best[wheel+3][station][scsector] > -1 && phcode_best[wheel+3][station][scsector] < 7) {
	      
	      histoTag = "DCC_TrackPosandTrig" + trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("Segment"), histoTag );
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(x_track);
	      histoTag = "DCC_TrackAngleandTrig"+ trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("Segment"), histoTag );
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(x_angle);
	      
	      // SM phi of the track vs phi of the trigger
	      histoTag = "DCC_PhitkvsPhitrig"+ trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("Segment"), histoTag );
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*ibest[wheel+3][station][scsector]).phi(),x_track);
	      
	      // SM hits of the track vs quality of the trigger
	      histoTag =  "DCC_HitstkvsQualtrig" + trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("Segment"), histoTag );
	     
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*ibest[wheel+3][station][scsector]).code(),(*track).degreesOfFreedom() );
	    }

	    if((*track).hasZed()) { // Theta component
	      if (thcode_best[wheel+3][station][scsector] > 0) {		
		histoTag = "DCC_TrackThetaPosandTrig" + trigsrc;
		if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		  bookHistos( dtChId, string("Segment"), histoTag );
		(digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(y_track);
		
		histoTag = "DCC_TrackThetaAngleandTrig" + trigsrc;
		if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		  bookHistos( dtChId, string("Segment"), histoTag );
		(digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(y_angle);		
	      }
	    }
	   
	  }
	  if ( parameters.getUntrackedParameter<bool>("process_ros", true) ) {	    
	    if (dduphcode_best[wheel+3][station][scsector] > -1 && dduphcode_best[wheel+3][station][scsector] < 7) {
	      
	      histoTag = "DDU_TrackPosandTrig" + trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("Segment"), histoTag );
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(x_track);

	      histoTag = "DDU_TrackAngleandTrig" + trigsrc;
	      if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		bookHistos( dtChId, string("Segment"), histoTag );             
	      (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(x_angle);

	      if((*track).hasZed()) { // Theta component
		if (dduthcode_best[wheel+3][station][scsector] > 0) {
	
		  histoTag = "DDU_TrackThetaPosandTrig" + trigsrc;
		  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		    bookHistos( dtChId, string("Segment"), histoTag );
		  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(y_track);
		  
		  histoTag = "DDU_TrackThetaAngleandTrig" + trigsrc;
		  if ((digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()))
		    bookHistos( dtChId, string("Segment"), histoTag );
       		  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill(y_angle);
		  
		}
	      }
	    }
	  }
	}
      }
    }  
  }
  
}

void DTLocalTriggerTask::bookHistos(const DTChamberId& dtCh, string folder, string histoTag) {

  int wh=dtCh.wheel();		
  int sc=dtCh.sector();	
  stringstream wheel; wheel << wh;	
  stringstream station; station << dtCh.station();	
  stringstream sector; sector << sc;	

  string histoType = histoTag.substr(4,histoTag.find("_",4)-4);

  if (debug)
    cout<<"[DTLocalTriggerTask]: booking"<<endl;

  if ( folder == "SectorGeneral") {   // Booking of sector quantities
    
    dbe->setCurrentFolder("DT/DTLocalTriggerTask/Wheel" + wheel.str() +
			  "/Sector" + sector.str() + "/" + folder );

    if (debug)
      cout << "[DTLocalTriggerTask]: folder "<< "DT/DTLocalTriggerTask/Wheel" << wheel.str() 
	   << "/Sector" << sector.str()  << "/" << folder << endl;
    
    string histoName = histoTag + "_W" + wheel.str() + "_Sec" + sector.str();

    if( histoType == "SCdata") {
      uint32_t indexScWh = 5*(sc-1) + (wh+3) ;    // wheel + sector identifier for specific histograms      
      (digiHistos[histoTag])[indexScWh] = 
	dbe->book1D(histoName,histoName,6,-0.5,5.5);
    }
    
  }
  else {    // Booking of station quantities
    
    dbe->setCurrentFolder("DT/DTLocalTriggerTask/Wheel" + wheel.str() +
			  "/Sector" + sector.str() +
			  "/Station" + station.str() + "/" + folder);

    string histoName = histoTag + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
    
    if (debug){
      cout << "[DTLocalTriggerTask]: folder " << "DT/DTLocalTriggerTask/Wheel" << wheel.str()
	   << "/Sector" << sector.str()
	   << "/Station"<< station.str() << "/" << folder << endl;
      cout << "[DTLocalTriggerTask]: histoType " << histoType << endl; 
      cout << "[DTLocalTriggerTask]: histoName " << histoName << endl;
    }
    
    if ( folder == "LocalTriggerPhi") {
      
      if( histoType == "BXvsQual"){
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,8,-0.5,7.5,101,-50.5,50.5);
	return ;
      }
      if( histoType == "TS2TagCodevsQual"){ 
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,8,-0.5,7.5,2,-0.5,1.5);  
	return ;
      }
      if( histoType == "QualvsPhirad"){
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,100,-1500.,1500.,8,-0.5,7.5);
	return ;
      }
      if( histoType == "QualvsPhibend") { 
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,100,-50.,50.,8,-0.5,7.5);
	return ;
      }
      if( histoType == "Flag1stvsBX") { 
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,101,-50.5,50.5,2,-0.5,1.5);
	return ;
      }
      if( histoType == "Flag1stvsQual") {
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,8,-0.5,7.5,2,-0.5,1.5);
	return ;
      }
      
    }
    else if ( folder == "LocalTriggerTheta")   {
      
      if( histoType == "PositionvsBX") {
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,101,-50.5,50.5,7,-0.5,6.5);
	return ;
      }
      if( histoType == "PositionvsCode") {
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,3,-0.5,2.5,6,-0.5,6.5);
	return ;
      }  
      if( histoType == "ThetaBXvsQual") {
 	(digiHistos[histoTag])[dtCh.rawId()] = 
 	  dbe->book2D(histoName,histoName,8,-0.5,7.5,60,-9.5,50.5);
      }

    }
    else if ( folder == "Segment")   {
      
      if( histoType == "TrackThetaPos") {
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book1D(histoName,histoName,40,-117.5,117.5);
	return ;
      }
      if( histoType == "TrackThetaAngle") {
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book1D(histoName,histoName,50,-2.,2.);
	return ;
      }      
      if( histoType == "TrackPos"){
	pair<float,float> range = phiRange(dtCh);
	int nbins = int((range.second - range.first)/15);
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book1D(histoName,histoName,nbins,range.first,range.second);
	return ;
      }
      if( histoType == "TrackAngle"){ 
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book1D(histoName,histoName,50,-2.,2.);
	return ;
      }
      if( histoType == "TrackPosandTrig"){ 
	pair<float,float> range = phiRange(dtCh);
	int nbins = int((range.second - range.first)/15);
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book1D(histoName,histoName,nbins,range.first,range.second);
	return ;
      }      
      if( histoType == "TrackAngleandTrig"){ 
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book1D(histoName,histoName,50,-2.,2.);
	return ;
      }
      if( histoType == "PhitkvsPhitrig"){ 
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,100,-1800.,1800.,100,-500.,500.);
	return ;
      }
      if( histoType == "HitstkvsQualtrig"){ 
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book2D(histoName,histoName,8,-0.5,7.5,10,0.5,10.5);
	return ;
      }
      if( histoType == "TrackThetaPosandTrig") {
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book1D(histoName,histoName,40,-117.5,117.5);
	return ;
      }
      if( histoType == "TrackThetaAngleandTrig") {
	(digiHistos[histoTag])[dtCh.rawId()] = 
	  dbe->book1D(histoName,histoName,50,-2.,2.);
	return ;
      }

    }
  }

}
//SM end

pair<float,float> DTLocalTriggerTask::phiRange(const DTChamberId& id){

  float min,max;
  int station = id.station();
  int sector  = id.sector(); 
  int wheel   = id.wheel();
  
  const DTLayer  *layer = muonGeom->layer(DTLayerId(id,1,1));
  DTTopology topo = layer->specificTopology();
  min = topo.wirePosition(topo.firstChannel());
  max = topo.wirePosition(topo.lastChannel());

  if (station == 4){
    
    const DTLayer *layer2;
    float lposx;
    
    if (sector == 4){
      layer2  = muonGeom->layer(DTLayerId(wheel,station,13,1,1));
      lposx = layer->toLocal(layer2->position()).x();
    }
    else if (sector == 10){
      layer2 = muonGeom->layer(DTLayerId(wheel,station,14,1,1));
      lposx = layer->toLocal(layer2->position()).x();
    }
    else
      return make_pair(min,max);
    
    DTTopology topo2 = layer2->specificTopology();

    if (lposx>0)
      max = lposx+topo2.wirePosition(topo2.lastChannel());
    else 
      min = lposx-topo2.wirePosition(topo2.firstChannel());
  }
  
  return make_pair(min,max);

}

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

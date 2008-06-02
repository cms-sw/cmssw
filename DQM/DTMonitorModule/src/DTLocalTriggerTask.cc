/*
 * \file DTLocalTriggerTask.cc
 * 
 * $Date: 2008/03/08 11:34:20 $
 * $Revision: 1.21 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "DQM/DTMonitorModule/interface/DTLocalTriggerTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// DT trigger
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"

// Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

//Root
#include"TH1.h"
#include"TAxis.h"

#include <sstream>
#include <iostream>
#include <fstream>


using namespace edm;
using namespace std;

DTLocalTriggerTask::DTLocalTriggerTask(const edm::ParameterSet& ps){
  
  debug = ps.getUntrackedParameter<bool>("debug", "false");

  if(debug)   
    cout<<"[DTLocalTriggerTask]: Constructor"<<endl;

  parameters = ps;
  loadDTTFMap();
  
  dbe = edm::Service<DQMStore>().operator->();

}


DTLocalTriggerTask::~DTLocalTriggerTask() {

  if(debug)
    cout << "[DTLocalTriggerTask]: analyzed " << nevents << " events" << endl;

}


void DTLocalTriggerTask::beginJob(const edm::EventSetup& context){

 if(debug)
    cout<<"[DTLocalTriggerTask]: BeginJob"<<endl;

 context.get<MuonGeometryRecord>().get(muonGeom);
 nevents = 0;

 if(parameters.getUntrackedParameter<bool>("staticBooking", true)) {  // Static histo booking
   
   vector<string> trigSources;
   if(parameters.getUntrackedParameter<bool>("localrun", true)) {
     trigSources.push_back("");
   }
   else {
     trigSources.push_back("_DTonly");
     trigSources.push_back("_NoDT");
     trigSources.push_back("_DTalso");
   }
   vector<string>::const_iterator trigSrcIt  = trigSources.begin();
   vector<string>::const_iterator trigSrcEnd = trigSources.end();
   
  
   for (;trigSrcIt!=trigSrcEnd;++trigSrcIt){
     for (int stat=1;stat<5;++stat){
       for (int wh=-2;wh<3;++wh){
	 for (int sect=1;sect<13;++sect){
	   DTChamberId dtChId(wh,stat,sect);

	   if (parameters.getUntrackedParameter<bool>("process_dcc", true)){ // DCC data
	   
	     bookHistos(dtChId,"LocalTriggerPhi","DCC_BXvsQual"+(*trigSrcIt));
	     bookHistos(dtChId,"LocalTriggerPhi","DCC_QualvsPhirad"+(*trigSrcIt));
	     bookHistos(dtChId,"LocalTriggerPhi","DCC_QualvsPhibend"+(*trigSrcIt));
	     // bookHistos(dtChId,"LocalTriggerPhi","DCC_Flag1stvsBX"+(*trigSrcIt));
	     bookHistos(dtChId,"LocalTriggerPhi","DCC_Flag1stvsQual"+(*trigSrcIt));
	     bookHistos(dtChId,"LocalTriggerPhi","DCC_BestQual"+(*trigSrcIt));
	     if (stat!=4){                                                   // theta view
	       bookHistos(dtChId,"LocalTriggerTheta","DCC_PositionvsBX"+(*trigSrcIt));
	       bookHistos(dtChId,"LocalTriggerTheta","DCC_PositionvsQual"+(*trigSrcIt));
	       bookHistos(dtChId,"LocalTriggerTheta","DCC_ThetaBXvsQual"+(*trigSrcIt));
	       bookHistos(dtChId,"LocalTriggerTheta","DCC_ThetaBestQual"+(*trigSrcIt));
	     }

	     if (parameters.getUntrackedParameter<bool>("process_seg", true)){ // DCC + Segemnt
	       bookHistos(dtChId,"Segment","DCC_PhitkvsPhitrig"+(*trigSrcIt));
	       bookHistos(dtChId,"Segment","DCC_PhibtkvsPhibtrig"+(*trigSrcIt));
	       bookHistos(dtChId,"Segment","DCC_HitstkvsQualtrig"+(*trigSrcIt));
	       bookHistos(dtChId,"Segment","DCC_TrackPosvsAngle"+(*trigSrcIt));
	       bookHistos(dtChId,"Segment","DCC_TrackPosvsAngleandTrig"+(*trigSrcIt));
	       bookHistos(dtChId,"Segment","DCC_TrackPosvsAngleandTrigHHHL"+(*trigSrcIt));
	       if(stat!=4){
		 bookHistos(dtChId,"Segment","DCC_TrackThetaPosvsAngle"+(*trigSrcIt)); // theta view
		 bookHistos(dtChId,"Segment","DCC_TrackThetaPosvsAngleandTrig"+(*trigSrcIt));
		 bookHistos(dtChId,"Segment","DCC_TrackThetaPosvsAngleandTrigH"+(*trigSrcIt));
	       }
	     }

	   }

	   if (parameters.getUntrackedParameter<bool>("process_ros", true)){ // DDU data
	   
	     bookHistos(dtChId,"LocalTriggerPhi","DDU_BXvsQual"+(*trigSrcIt));
	     // bookHistos(dtChId,"LocalTriggerPhi","DDU_Flag1stvsBX"+(*trigSrcIt));
	     bookHistos(dtChId,"LocalTriggerPhi","DDU_Flag1stvsQual"+(*trigSrcIt));
	     bookHistos(dtChId,"LocalTriggerPhi","DDU_BestQual"+(*trigSrcIt));
	     if(stat!=4){                                                    // theta view
	       bookHistos(dtChId,"LocalTriggerTheta","DDU_ThetaBXvsQual"+(*trigSrcIt));
	       bookHistos(dtChId,"LocalTriggerTheta","DDU_ThetaBestQual"+(*trigSrcIt));
	     }
	   
	     if (parameters.getUntrackedParameter<bool>("process_seg", true)){ // DDU + Segment
	       bookHistos(dtChId,"Segment","DDU_HitstkvsQualtrig"+(*trigSrcIt));
	       bookHistos(dtChId,"Segment","DDU_TrackPosvsAngle"+(*trigSrcIt));
	       bookHistos(dtChId,"Segment","DDU_TrackPosvsAngleandTrig"+(*trigSrcIt));
	       bookHistos(dtChId,"Segment","DDU_TrackPosvsAngleandTrigHHHL"+(*trigSrcIt));
	       if(stat!=4){
		 bookHistos(dtChId,"Segment","DDU_TrackThetaPosvsAngle"+(*trigSrcIt)); // theta view
		 bookHistos(dtChId,"Segment","DDU_TrackThetaPosvsAngleandTrig"+(*trigSrcIt));
		 bookHistos(dtChId,"Segment","DDU_TrackThetaPosvsAngleandTrigH"+(*trigSrcIt));
	       }
	     }
	   
	   }

	   if (parameters.getUntrackedParameter<bool>("process_dcc", true) &&
	       parameters.getUntrackedParameter<bool>("process_ros", true)){ // DCC+DDU data
	     bookHistos(dtChId,"LocalTriggerPhi","COM_QualDDUvsQualDCC"+(*trigSrcIt));
	   }
	   
	 }
       }
     }
   }// end of loop

 }

}

 void DTLocalTriggerTask::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

  if(debug)
    cout<<"[DTLocalTriggerTask]: Begin of LS transition"<<endl;
  
  if(lumiSeg.id().luminosityBlock()%parameters.getUntrackedParameter<int>("ResetCycle", 3) == 0) {
    for(map<uint32_t, map<string, MonitorElement*> > ::const_iterator histo = digiHistos.begin();
	histo != digiHistos.end();
	histo++) {
      for(map<string, MonitorElement*> ::const_iterator ht = (*histo).second.begin();
	  ht != (*histo).second.end();
	  ht++) {
	(*ht).second->Reset();
      }
    }
  }
  
}


void DTLocalTriggerTask::endJob(){

  cout << "[DTLocalTriggerTask]: analyzed " << nevents << " events" << endl;
  dbe->rmdir("DT/DTLocalTriggerTask");

}


void DTLocalTriggerTask::analyze(const edm::Event& e, const edm::EventSetup& c){
  
  string dcc_label  = parameters.getUntrackedParameter<string>("dcc_label", "dttpgprod");
  string ros_label  = parameters.getUntrackedParameter<string>("ros_label", "dtunpacker");
  string seg_label  = parameters.getUntrackedParameter<string>("seg_label", "dt4DSegments");

  if (!nevents){

    edm::Handle<L1MuDTChambPhContainer> l1DTTPGPh;
    e.getByLabel(dcc_label, l1DTTPGPh);
    edm::Handle<L1MuDTChambThContainer> l1DTTPGTh;
    e.getByLabel(dcc_label, l1DTTPGTh);
    useDCC = (l1DTTPGPh.isValid() || l1DTTPGTh.isValid()) && parameters.getUntrackedParameter<bool>("process_dcc", true) ;

    Handle<DTLocalTriggerCollection> l1DDUTrigs;
    e.getByLabel(ros_label,l1DDUTrigs);
    useDDU = l1DDUTrigs.isValid() && parameters.getUntrackedParameter<bool>("process_ros", true) ;

    Handle<DTRecSegment4DCollection> all4DSegments;
    e.getByLabel(seg_label, all4DSegments);  
    useSEG = all4DSegments.isValid() && parameters.getUntrackedParameter<bool>("process_seg", true) ;
    
  }

  nevents++;
    
  triggerSource(e);  

  if ( useDCC ) {   
    edm::Handle<L1MuDTChambPhContainer> l1DTTPGPh;
    e.getByLabel(dcc_label,l1DTTPGPh);
    vector<L1MuDTChambPhDigi>*  l1PhTrig = l1DTTPGPh->getContainer();

    edm::Handle<L1MuDTChambThContainer> l1DTTPGTh;
    e.getByLabel(dcc_label,l1DTTPGTh);
    vector<L1MuDTChambThDigi>*  l1ThTrig = l1DTTPGTh->getContainer();

    runDCCAnalysis(l1PhTrig,l1ThTrig);
  }  
  if ( useDDU ) {
    Handle<DTLocalTriggerCollection> l1DDUTrigs;
    e.getByLabel(ros_label,l1DDUTrigs);

    runDDUAnalysis(l1DDUTrigs);
  }
  if ( useSEG ) {
    Handle<DTRecSegment4DCollection> segments4D;
    e.getByLabel(seg_label, segments4D);  
    
    runSegmentAnalysis(segments4D);
  } 
  if ( useDCC && useDDU ) {
    runDDUvsDCCAnalysis(trigsrc);
  }

}


void DTLocalTriggerTask::bookHistos(const DTChamberId& dtCh, string folder, string histoTag) {
  
  int wh=dtCh.wheel();		
  int sc=dtCh.sector();	
  stringstream wheel; wheel << wh;	
  stringstream station; station << dtCh.station();	
  stringstream sector; sector << sc;	

  double minBX=0;
  double maxBX=0;
  int  rangeBX=0;

  string histoType = histoTag.substr(4,histoTag.find("_",4)-4);

  dbe->setCurrentFolder("DT/DTLocalTriggerTask/Wheel" + wheel.str() +
			"/Sector" + sector.str() +
			"/Station" + station.str() + "/" + folder);

  string histoName = histoTag + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
    
  if (debug){
    cout << "[DTLocalTriggerTask]: booking " << "DT/DTLocalTriggerTask/Wheel" << wheel.str()
	 << "/Sector" << sector.str()
	 << "/Station"<< station.str() << "/" << folder << "/" << histoName << endl;
  }
    
  if (histoType.find("BX") != string::npos){
    if (histoTag.substr(0,3) == "DCC"){
      minBX= parameters.getUntrackedParameter<int>("minBXDCC",-2) - 0.5;
      maxBX= parameters.getUntrackedParameter<int>("maxBXDCC",2) + 0.5;
    }
    else {
      minBX= parameters.getUntrackedParameter<int>("minBXDDU",0) - 0.5;
      maxBX= parameters.getUntrackedParameter<int>("maxBXDDU",20) + 0.5;
    }
    rangeBX = (int)(maxBX-minBX);
  }
    
  if ( folder == "LocalTriggerPhi") {
      
    if( histoType == "BXvsQual" ){
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"BX vs trigger quality",8,-0.5,7.5,rangeBX,minBX,maxBX);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],1);
      return ;
    }
    if( histoType == "BestQual" ){
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book1D(histoName,"Trigger quality of best primitives",7,-0.5,6.5);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],1);
      return ;
    }
    if( histoType == "QualvsPhirad" ){
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"Trigger quality vs local position",100,-500.,500.,8,-0.5,7.5);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],2);
      return ;
    }
    if( histoType == "QualvsPhibend" ) { 
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"Trigger quality vs local direction",200,-40.,40.,8,-0.5,7.5);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],2);      
      return ;
    }
//     if( histoType == "Flag1stvsBX" ) { 
//       (digiHistos[dtCh.rawId()])[histoTag] = 
// 	dbe->book2D(histoName,"1st/2nd trig flag vs BX",rangeBX,minBX,maxBX,2,-0.5,1.5);
//       return ;
//     }
    if( histoType == "Flag1stvsQual" ) {
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"1st/2nd trig flag vs quality",8,-0.5,7.5,2,-0.5,1.5);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],1);
      return ;
    }
    if( histoType == "QualDDUvsQualDCC" ){
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"DDU quality vs DCC quality",9,-1.5,7.5,9,-1.5,7.5);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],1);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],2);      
      return ;
    }    
      
  }
  else if ( folder == "LocalTriggerTheta")   {
      
    if( histoType == "PositionvsBX" ) {
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"Theta trigger position vs BX",rangeBX,minBX,maxBX,7,-0.5,6.5);
      return ;
    }
    if( histoType == "PositionvsQual" ) {
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"Theta trigger position vs quality",8,-0.5,7.5,6,-0.5,6.5);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],1);
      return ;
    }  
    if( histoType == "ThetaBXvsQual" ) {
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"BX vs trigger quality",8,-0.5,7.5,rangeBX,minBX,maxBX);
    }
    if( histoType == "ThetaBestQual" ){
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book1D(histoName,"Trigger quality of best primitives (theta)",8,-0.5,7.5);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],1);
      return ;
    }

  }
  else if ( folder == "Segment")   {
      
    if( histoType.find("TrackThetaPosvsAngle" ) == 0 ) {

      string histoLabel = "Position vs Angle (theta)";
      if (histoType.find("andTrigH") != string::npos) histoLabel += " for H triggers";
      else if (histoType.find("andTrig") != string::npos) histoLabel += " for triggers";

      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,histoLabel,16,-40.,40.,20,-117.5,117.5);
      return ;
    }
    if( histoType.find("TrackPosvsAngle") == 0 ){

      pair<float,float> range = phiRange(dtCh);
      int nbins = int((range.second - range.first)/15);

      string histoLabel = "Position vs Angle (phi)";
      if (histoType.find("andTrigHHHL")  != string::npos) histoLabel += " for HH/HL triggers";
      else if (histoType.find("andTrig") != string::npos) histoLabel += " for triggers";

      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,histoLabel,16,-40.,40.,nbins,range.first,range.second);
      return ;
    }
    if( histoType == "PhitkvsPhitrig" ){ 
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"Local position: segment vs trigger",100,-500.,500.,100,-500.,500.);
      return ;
    }
    if( histoType == "PhibtkvsPhibtrig" ){ 
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"Local direction : segment vs trigger",200,-40.,40.,200,-40.,40.);
      return ;
    }
    if( histoType == "HitstkvsQualtrig" ){ 
      (digiHistos[dtCh.rawId()])[histoTag] = 
	dbe->book2D(histoName,"Segment hits (phi) vs trigger quality",8,-0.5,7.5,10,0.5,10.5);
      setQLabels((digiHistos[dtCh.rawId()])[histoTag],1);
      return ;
    }

  }

}

void DTLocalTriggerTask::runDCCAnalysis( std::vector<L1MuDTChambPhDigi>* phTrigs, 
					 std::vector<L1MuDTChambThDigi>* thTrigs ){

  string histoType ;
  string histoTag ;

  // define best quality trigger segment (phi and theta)  
  // in any station start from 1 and zero is kept empty
  for (int i=0;i<5;++i){
    for (int j=0;j<6;++j){
      for (int k=0;k<13;++k){
	phcode_best[j][i][k] = -1;
	thcode_best[j][i][k] = -1;
      }
    }
  }

  vector<L1MuDTChambPhDigi>::const_iterator iph  = phTrigs->begin();
  vector<L1MuDTChambPhDigi>::const_iterator iphe = phTrigs->end();
  for(; iph !=iphe ; ++iph) {

    int phwheel = iph->whNum();
    int phsec   = iph->scNum() + 1; // SM The track finder goes from 0 to 11. I need them from 1 to 12 !!!!!
    int phst    = iph->stNum();
    int phbx    = iph->bxNum();
    int phcode  = iph->code();
    int phi1st  = iph->Ts2Tag();
    int phphi   = iph->phi();
    int phphiB  = iph->phiB();

    correctMapping(phwheel,phsec);

    if(phcode>phcode_best[phwheel+3][phst][phsec] && phcode<7) {
      phcode_best[phwheel+3][phst][phsec]=phcode; 
      iphbest[phwheel+3][phst][phsec] = &(*iph);
    }
      
    DTChamberId dtChId(phwheel,phst,phsec);
      
    float x     = phi2Pos(dtChId,phphi);
    float angle = phib2Ang(dtChId,phphiB,phphi);
    uint32_t indexCh = dtChId.rawId();
    
    map<string, MonitorElement*> &innerME = digiHistos[indexCh];
    if (innerME.find("DCC_BXvsQual"+trigsrc) == innerME.end()){
      bookHistos(dtChId,"LocalTriggerPhi","DCC_BXvsQual"+trigsrc);
      bookHistos(dtChId,"LocalTriggerPhi","DCC_QualvsPhirad"+trigsrc);
      bookHistos(dtChId,"LocalTriggerPhi","DCC_QualvsPhibend"+trigsrc);
//       bookHistos(dtChId,"LocalTriggerPhi","DCC_Flag1stvsBX"+trigsrc);
      bookHistos(dtChId,"LocalTriggerPhi","DCC_Flag1stvsQual"+trigsrc);
    }

    innerME.find("DCC_BXvsQual"+trigsrc)->second->Fill(phcode,phbx-phi1st);    // SM BX vs Qual Phi view (1st tracks) 
    innerME.find("DCC_QualvsPhirad"+trigsrc)->second->Fill(x,phcode);          // SM Qual vs radial angle Phi view
    innerME.find("DCC_QualvsPhibend"+trigsrc)->second->Fill(angle,phcode);     // SM Qual vs bending Phi view
//     innerME.find("DCC_Flag1stvsBX"+trigsrc)->second->Fill(phbx,phi1st);        // SM BX vs 1st/2nd track flag Phi view
    innerME.find("DCC_Flag1stvsQual"+trigsrc)->second->Fill(phcode,phi1st);    // SM Qual 1st/2nd track flag Phi view
      
  } 


  int thcode[7];

  vector<L1MuDTChambThDigi>::const_iterator ith  = thTrigs->begin();
  vector<L1MuDTChambThDigi>::const_iterator ithe = thTrigs->end();
  for(; ith != ithe; ++ith) {
    int thwheel = ith->whNum();
    int thsec   = ith->scNum() + 1; // SM The track finder goes from 0 to 11. I need them from 1 to 12 !!!!!
    int thst    = ith->stNum();
    int thbx    = ith->bxNum();
    correctMapping(thwheel,thsec);
      
    for (int pos=0; pos<7; pos++) {
      thcode[pos] = ith->code(pos);

      if(thcode[pos]>thcode_best[thwheel+3][thst][thsec] ) {
	thcode_best[thwheel+3][thst][thsec]=thcode[pos]; 
	ithbest[thwheel+3][thst][thsec] = &(*ith);
      }
    } 
      
    DTChamberId dtChId(thwheel,thst,thsec);
    uint32_t indexCh = dtChId.rawId();   

    map<string, MonitorElement*> &innerME = digiHistos[indexCh];
    if (innerME.find("DCC_PositionvsBX"+trigsrc) == innerME.end()){
      bookHistos(dtChId,"LocalTriggerTheta","DCC_PositionvsBX"+trigsrc);
      bookHistos(dtChId,"LocalTriggerTheta","DCC_PositionvsQual"+trigsrc);
      bookHistos(dtChId,"LocalTriggerTheta","DCC_ThetaBXvsQual"+trigsrc);
    }

    for (int pos=0; pos<7; pos++) //SM fill position for non zero position bit in theta view
      if(thcode[pos]>0){
	int thqual = (thcode[pos]/2)*2+1;
	innerME.find("DCC_PositionvsBX"+trigsrc)->second->Fill(thbx,pos);          // SM BX vs Position Theta view
	innerME.find("DCC_PositionvsQual"+trigsrc)->second->Fill(thqual,pos);      // SM Code vs Position Theta view
	innerME.find("DCC_ThetaBXvsQual"+trigsrc)->second->Fill(thqual,thbx);      // SM BX vs Code Theta view
      }

  }

  
  // Fill Quality plots with best DCC triggers in phi & theta
  for (int st=1;st<5;++st){
    for (int wh=-2;wh<3;++wh){
      for (int sc=1;sc<13;++sc){
	if (phcode_best[wh+3][st][sc]>-1 && phcode_best[wh+3][st][sc]<7){
	  DTChamberId id(wh,st,sc);
	  uint32_t indexCh = id.rawId();
	  map<string, MonitorElement*> &innerME = digiHistos[indexCh];
	  if (innerME.find("DCC_BestQual"+trigsrc) == innerME.end())
	    bookHistos(id,"LocalTriggerPhi","DCC_BestQual"+trigsrc);
	  innerME.find("DCC_BestQual"+trigsrc)->second->Fill(phcode_best[wh+3][st][sc]);  // CB Best Qual Trigger Phi view
	}
	if (thcode_best[wh+3][st][sc]>0){
	  DTChamberId id(wh,st,sc);
	  uint32_t indexCh = id.rawId();
	  map<string, MonitorElement*> &innerME = digiHistos[indexCh];
	  if (innerME.find("DCC_ThetaBestQual"+trigsrc) == innerME.end())
	    bookHistos(id,"LocalTriggerTheta","DCC_ThetaBestQual"+trigsrc);
	   innerME.find("DCC_ThetaBestQual"+trigsrc)->second->Fill(thcode_best[wh+3][st][sc]); // CB Best Qual Trigger Phi view
	}
      }
    }
  }

}

void DTLocalTriggerTask::runDDUAnalysis(Handle<DTLocalTriggerCollection>& trigsDDU){
    
  DTLocalTriggerCollection::DigiRangeIterator detUnitIt;
    
  for (int i=0;i<5;++i){
    for (int j=0;j<6;++j){
      for (int k=0;k<13;++k){
	dduphcode_best[j][i][k] = -1;
	dduthcode_best[j][i][k] = -1;
      }
    }
  }

  for (detUnitIt=trigsDDU->begin();
       detUnitIt!=trigsDDU->end();
       ++detUnitIt){
      
    const DTChamberId& id = (*detUnitIt).first;
    const DTLocalTriggerCollection::Range& range = (*detUnitIt).second;
    uint32_t indexCh = id.rawId();  
    map<string, MonitorElement*> &innerME = digiHistos[indexCh];
      
    int wh = id.wheel();
    int sec = id.sector();
    int st = id.station();

    for (DTLocalTriggerCollection::const_iterator trigIt = range.first;
	 trigIt!=range.second;
	 ++trigIt){
	
      int bx = trigIt->bx();
      int quality = trigIt->quality();
      int thqual = trigIt->trTheta();
      int flag1st = trigIt->secondTrack() ? 1 : 0;

      // check if SC data exist: fill for any trigger
      if( quality>-1 && quality<7 ) {	  // it is a phi trigger
	
	if(quality>dduphcode_best[wh+3][st][sec]) {
	  dduphcode_best[wh+3][st][sec]=quality;
	  iphbestddu[wh+3][st][sec] = &(*trigIt);
	}
	
	if (innerME.find("DDU_BXvsQual"+trigsrc) == innerME.end()){
	  bookHistos(id,"LocalTriggerPhi","DDU_BXvsQual"+trigsrc);
// 	  bookHistos(id,"LocalTriggerPhi","DDU_Flag1stvsBX"+trigsrc);
	  bookHistos(id,"LocalTriggerPhi","DDU_Flag1stvsQual"+trigsrc);
	}

	innerME.find("DDU_BXvsQual"+trigsrc)->second->Fill(quality,bx-flag1st);     // SM BX vs Qual Phi view	
// 	innerME.find("DDU_Flag1stvsBX"+trigsrc)->second->Fill(bx,flag1st);          // SM 1st/2nd track vs BX flag Phi view
	innerME.find("DDU_Flag1stvsQual"+trigsrc)->second->Fill(quality,flag1st); // SM Quality vs 1st/2nd track flag Phi view
	
      }
      if( thqual>0) {  // it is a theta trigger
	
	if(thqual>dduthcode_best[wh+3][st][sec] ) {
	  dduthcode_best[wh+3][st][sec]=thqual; 
	}
	if (innerME.find("DDU_ThetaBXvsQual"+trigsrc) == innerME.end())
	  bookHistos(id,"LocalTriggerTheta","DDU_ThetaBXvsQual"+trigsrc);
	innerME.find("DDU_ThetaBXvsQual"+trigsrc)->second->Fill(thqual,bx);     // SM BX vs Qual Theta view
	
      }
    }
    
    // Fill Quality plots with best ddu triggers in phi & theta
    if (dduphcode_best[wh+3][st][sec]>-1 &&
	dduphcode_best[wh+3][st][sec]<7){
      if (innerME.find("DDU_BestQual"+trigsrc) == innerME.end())
	bookHistos(id,"LocalTriggerPhi","DDU_BestQual"+trigsrc);
      innerME.find("DDU_BestQual"+trigsrc)->second->Fill(dduphcode_best[wh+3][st][sec]);  // CB Best Qual Trigger Phi view
    }
    if (dduthcode_best[wh][st][sec]>0){
      if (innerME.find("DDU_ThetaBestQual"+trigsrc) == innerME.end())
	bookHistos(id,"LocalTriggerTheta","DDU_ThetaBestQual"+trigsrc);
      innerME.find("DDU_ThetaBestQual"+trigsrc)->second->Fill(dduthcode_best[wh+3][st][sec]); // CB Best Qual Trigger Theta view
    }  
  }

}


void DTLocalTriggerTask::runSegmentAnalysis(Handle<DTRecSegment4DCollection>& segments4D){    

  DTRecSegment4DCollection::const_iterator track;

  // Find best tracks & good tracks
  memset(track_ok,false,450*sizeof(bool));

  DTRecSegment4DCollection::id_iterator chamberId;
  vector<const DTRecSegment4D*> best4DSegments;

  // Preliminary loop finds best 4D Segment and high quality ones
  for (chamberId = segments4D->id_begin(); chamberId != segments4D->id_end(); ++chamberId){

    DTRecSegment4DCollection::range  range = segments4D->get(*chamberId);
    const DTRecSegment4D* tmpBest=0;
    int tmpdof = 0;
    int dof = 0;

    for ( track = range.first; track != range.second; ++track){

      if( (*track).hasPhi() ) {
	
	dof = (*track).phiSegment()->degreesOfFreedom();
	if ( dof>tmpdof ){
	  tmpBest = &(*track);
	  tmpdof = dof;
	
	  int wheel = (*track).chamberId().wheel();
	  int sector = (*track).chamberId().sector();
	  int station = (*track).chamberId().station();
	  if (sector==13){
	    sector=4;
	  }
	  else if (sector==14){
	    sector=10;
	  }
	  track_ok[wheel+3][station][sector] = (!track_ok[wheel+3][station][sector] && dof>=2);
	}

      }
    }
    if (tmpBest) best4DSegments.push_back(tmpBest);
  }
    
  vector<const DTRecSegment4D*>::const_iterator btrack;

  for ( btrack = best4DSegments.begin(); btrack != best4DSegments.end(); ++btrack ){

    if( (*btrack)->hasPhi() ) { // Phi component
	
      int wheel = (*btrack)->chamberId().wheel();
      int station = (*btrack)->chamberId().station();
      int scsector = 0;
      float x_track, y_track, x_angle, y_angle;
      computeCoordinates((*btrack),scsector,x_track,x_angle,y_track,y_angle);
      int nHitsPhi = (*btrack)->phiSegment()->degreesOfFreedom()+2;
	
      DTChamberId dtChId(wheel,station,scsector);  // get chamber for histograms
      uint32_t indexCh = dtChId.rawId(); 
      map<string, MonitorElement*> &innerME = digiHistos[indexCh];


      if (useDDU && 
	  dduphcode_best[wheel+3][station][scsector] > -1 && 
	  dduphcode_best[wheel+3][station][scsector] < 7 ) {

	// SM hits of the track vs quality of the trigger
	if (innerME.find("DDU_HitstkvsQualtrig"+trigsrc) == innerME.end())
	  bookHistos(dtChId,"Segment","DDU_HitstkvsQualtrig"+trigsrc);
	innerME.find("DDU_HitstkvsQualtrig"+trigsrc)->second->Fill(dduphcode_best[wheel+3][station][scsector],nHitsPhi);

      }
	
      if (useDCC &&
	  phcode_best[wheel+3][station][scsector] > -1 && 
	  phcode_best[wheel+3][station][scsector] < 7 ) {
	    
	int phphi = iphbest[wheel+3][station][scsector]->phi();
	float x_trigger = phi2Pos(dtChId,phphi);
	float angle_trigger = phib2Ang(dtChId,iphbest[wheel+3][station][scsector]->phiB(),phphi);
	    
	if (innerME.find("DCC_HitstkvsQualtrig"+trigsrc) == innerME.end()){
	  bookHistos(dtChId,"Segment","DCC_PhitkvsPhitrig"+trigsrc);
	  bookHistos(dtChId,"Segment","DCC_PhibtkvsPhibtrig"+trigsrc);
	  bookHistos(dtChId,"Segment","DCC_HitstkvsQualtrig"+trigsrc);
	}

	innerME.find("DCC_PhitkvsPhitrig"+trigsrc)->second->Fill(x_trigger,x_track);
	innerME.find("DCC_PhibtkvsPhibtrig"+trigsrc)->second->Fill(angle_trigger,x_angle);
	innerME.find("DCC_HitstkvsQualtrig"+trigsrc)->second->Fill(iphbest[wheel+3][station][scsector]->code(),nHitsPhi);

      }

      
      if (useDCC) {
	  
	// check for triggers elsewhere in the sector
	bool trigFlagDCC =false;
	for (int ist=1; ist<5; ist++){
	  if (ist!=station &&
	      phcode_best[wheel+3][ist][scsector]>=2 && 
	      phcode_best[wheel+3][ist][scsector]<7 &&
	      track_ok[wheel+3][ist][scsector]==true){
	    trigFlagDCC = true;
	    break;
	  }
	}
	  
	if (trigFlagDCC && fabs(x_angle)<40. && nHitsPhi>=7){

	  if (innerME.find("DCC_TrackPosvsAngle"+trigsrc) == innerME.end()){
	    bookHistos(dtChId,"Segment","DCC_TrackPosvsAngle"+trigsrc);
	    bookHistos(dtChId,"Segment","DCC_TrackPosvsAngleandTrig"+trigsrc);
	    bookHistos(dtChId,"Segment","DCC_TrackPosvsAngleandTrigHHHL"+trigsrc);
	  }
	    
	  // position vs angle of track for reconstruced tracks (denom. for trigger efficiency)
	  innerME.find("DCC_TrackPosvsAngle"+trigsrc)->second->Fill(x_angle,x_track);	  
	  if (phcode_best[wheel+3][station][scsector] >= 2 && phcode_best[wheel+3][station][scsector] < 7) {
	    innerME.find("DCC_TrackPosvsAngleandTrig"+trigsrc)->second->Fill(x_angle,x_track);	  
	    if (phcode_best[wheel+3][station][scsector] > 4){  //HH & HL Triggers
	      innerME.find("DCC_TrackPosvsAngleandTrigHHHL"+trigsrc)->second->Fill(x_angle,x_track);	  
	    }	      
	  }

	}
	  
	if ((*btrack)->hasZed() && trigFlagDCC && fabs(y_angle)<40. && (*btrack)->zSegment()->degreesOfFreedom()>=1){

	  if (innerME.find("DCC_TrackThetaPosvsAngle"+trigsrc) == innerME.end()){
	    bookHistos(dtChId,"Segment","DCC_TrackThetaPosvsAngle"+trigsrc);
	    bookHistos(dtChId,"Segment","DCC_TrackThetaPosvsAngleandTrig"+trigsrc);
	    bookHistos(dtChId,"Segment","DCC_TrackThetaPosvsAngleandTrigH"+trigsrc);
	  }

	  // position va angle of track for reconstruced tracks (denom. for trigger efficiency) along theta direction
	  innerME.find("DCC_TrackThetaPosvsAngle"+trigsrc)->second->Fill(y_angle,y_track);
	  if (thcode_best[wheel+3][station][scsector] > 0) {		
	    innerME.find("DCC_TrackThetaPosvsAngleandTrig"+trigsrc)->second->Fill(y_angle,y_track);
	    if (thcode_best[wheel+3][station][scsector] == 3) {
	      innerME.find("DCC_TrackThetaPosvsAngleH"+trigsrc)->second->Fill(y_angle,y_track);
	    }		 
	  }

	}  
      }

      if (useDDU) {
	  
	// check for triggers elsewhere in the sector
	bool trigFlagDDU =false;
	for (int ist=1; ist<5; ist++){
	  if (ist!=station &&
	      dduphcode_best[wheel+3][ist][scsector]>=2 && 
	      dduphcode_best[wheel+3][ist][scsector]<7 &&
	      track_ok[wheel+3][ist][scsector]==true){
	    trigFlagDDU = true;
	    break;
	  }
	}

	if (trigFlagDDU && fabs(x_angle)<40. && nHitsPhi>=7){

	  if (innerME.find("DDU_TrackPosvsAngle"+trigsrc) == innerME.end()){
	    bookHistos(dtChId,"Segment","DDU_TrackPosvsAngle"+trigsrc);
	    bookHistos(dtChId,"Segment","DDU_TrackPosvsAngleandTrig"+trigsrc);
	    bookHistos(dtChId,"Segment","DDU_TrackPosvsAngleandTrigHHHL"+trigsrc);
	  }

	  // position vs angle of track for reconstruced tracks (denom. for trigger efficiency)
	  innerME.find("DDU_TrackPosvsAngle"+trigsrc)->second->Fill(x_angle,x_track);
	  if (dduphcode_best[wheel+3][station][scsector] >= 2 && dduphcode_best[wheel+3][station][scsector] < 7) {
	  innerME.find("DDU_TrackPosvsAngleandTrig"+trigsrc)->second->Fill(x_angle,x_track);
	    if (dduphcode_best[wheel+3][station][scsector] > 4){  //HH & HL Triggers
	      innerME.find("DDU_TrackPosvsAngleandTrigHHHL"+trigsrc)->second->Fill(x_angle,x_track);	  
	    }	      
	  }

	}
	  
	if ((*btrack)->hasZed() && trigFlagDDU && fabs(y_angle)<40. && (*btrack)->zSegment()->degreesOfFreedom()>=1){

	  if (innerME.find("DDU_TrackThetaPosvsAngle"+trigsrc) == innerME.end()){
	    bookHistos(dtChId,"Segment","DDU_TrackThetaPosvsAngle"+trigsrc);
	    bookHistos(dtChId,"Segment","DDU_TrackThetaPosvsAngleandTrig"+trigsrc);
	    bookHistos(dtChId,"Segment","DDU_TrackThetaPosvsAngleandTrigH"+trigsrc);
	  }

	  // position va angle of track for reconstruced tracks (denom. for trigger efficiency) along theta direction
	  innerME.find("DDU_TrackThetaPosvsAngle"+trigsrc)->second->Fill(y_angle,y_track);
	  if (dduthcode_best[wheel+3][station][scsector] > 0) {		
	    innerME.find("DDU_TrackThetaPosvsAngleandTrig"+trigsrc)->second->Fill(y_angle,y_track);
	    if (dduthcode_best[wheel+3][station][scsector] == 3) {
	      innerME.find("DDU_TrackThetaPosvsAngleandTrigH"+trigsrc)->second->Fill(y_angle,y_track);
	    }		 
	  }

	}  
      }
    }
  } 

}


void DTLocalTriggerTask::runDDUvsDCCAnalysis(string& trigsrc){

  string histoType ;
  string histoTag ;

  for (int st=1;st<5;++st){
    for (int wh=-2;wh<3;++wh){
      for (int sc=1;sc<13;++sc){
	if ( (phcode_best[wh+3][st][sc]>-1 && phcode_best[wh+3][st][sc]<7) ||
	     (dduphcode_best[wh+3][st][sc]>-1 && dduphcode_best[wh+3][st][sc]<7) ){
	  DTChamberId id(wh,st,sc);
	  uint32_t indexCh = id.rawId();
	  map<string, MonitorElement*> &innerME = digiHistos[indexCh];
	  if (innerME.find("COM_QualDDUvsQualDCC"+trigsrc) == innerME.end())
	    bookHistos(id,"LocalTriggerPhi","COM_QualDDUvsQualDCC"+trigsrc);
	  innerME.find("COM_QualDDUvsQualDCC"+trigsrc)->second->Fill(phcode_best[wh+3][st][sc],dduphcode_best[wh+3][st][sc]);
	}
      }
    }
  }

}


void DTLocalTriggerTask::computeCoordinates(const DTRecSegment4D* track, 
					    int& scsector, 
					    float& phpos, 
					    float& phdir, 
					    float& zpos, 
					    float& zdir){


  float xcenter;	
  LocalPoint lpos;
  int wheel = track->chamberId().wheel();          
  int sector = track->chamberId().sector();
  int station = track->chamberId().station();
  const DTChamber* chamb;
  const DTChamber* scchamb;
  phdir = atan(track->localDirection().x()/ track->localDirection().z())/Geom::pi()*180.;
  zdir = atan(track->localDirection().y()/ track->localDirection().z())/Geom::pi()*180.;


  if (station == 4){
    switch (sector) {
    case 4:
      scsector = 4;
      chamb   = muonGeom->chamber(DTChamberId(wheel,station,13));
      scchamb = muonGeom->chamber(DTChamberId(wheel,station,4));
      xcenter = scchamb->toLocal(chamb->position()).x()*.5;
      phpos = track->localPosition().x()-xcenter;
      zpos  = track->localPosition().y();
      break;
    case 10:
      scsector = 10;
      chamb   = muonGeom->chamber(DTChamberId(wheel,station,14));
      scchamb = muonGeom->chamber(DTChamberId(wheel,station,10));
      xcenter = scchamb->toLocal(chamb->position()).x()*.5;
      phpos = track->localPosition().x()-xcenter;
      zpos = track->localPosition().y();
      break;
    case 13:
      scsector = 4;
      chamb   = muonGeom->chamber(DTChamberId(wheel,station,sector));
      scchamb = muonGeom->chamber(DTChamberId(wheel,station,scsector));
      lpos = scchamb->toLocal(chamb->toGlobal(track->localPosition()));
      xcenter = scchamb->toLocal(chamb->position()).x()*.5;
      phpos = lpos.x()-xcenter;
      zpos  = lpos.y();
      break;
    case 14:
      scsector = 10;
      chamb   = muonGeom->chamber(DTChamberId(wheel,station,sector));
      scchamb = muonGeom->chamber(DTChamberId(wheel,station,scsector));
      lpos = scchamb->toLocal(chamb->toGlobal(track->localPosition()));
      xcenter = scchamb->toLocal(chamb->position()).x()*.5;
      phpos = lpos.x()-xcenter;
      zpos  = lpos.y();
      break;
    default:
      scsector = sector;
      phpos = track->localPosition().x();
      zpos  = track->localPosition().y();
    }
  }
  else {
    scsector = sector;
    phpos = track->localPosition().x();
    zpos  = track->localPosition().y();
  }

}


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

    if (lposx>0){
      max =  lposx*.5 + topo2.wirePosition(topo2.lastChannel());
      min -= lposx*.5;
    }
    else{ 
      min =  lposx*.5 + topo2.wirePosition(topo2.firstChannel());
      max -= lposx*.5;
    }
  }

  return make_pair(min,max);
 
}

float DTLocalTriggerTask::phi2Pos(const DTChamberId & id, int phi){

  float phin = (id.sector()-1)*Geom::pi()/6;
  GlobalPoint gpos = muonGeom->chamber(id)->position();
  float deltaphi =  gpos.phi()-phin;

  if (id.station() == 4 && ( id.sector() == 4 || id.sector() == 10))
    deltaphi = 0;

  float x = (tan(phi/4096.)-tan(deltaphi))*gpos.perp()*cos(deltaphi);
  
  if (id.wheel()>0 || (id.wheel()==0 && id.sector()%4>1)) 
    x= -x;

  return x;

}


float DTLocalTriggerTask::phib2Ang(const DTChamberId & id, int phib, double phi){
  
  float fphi = phib/512.+phi/4096.;//+(id.sector()-4)*Geom::pi()/6.;
  fphi *= 180./Geom::pi();

  if (id.wheel()<0 || (id.wheel()==0 && id.sector()%4<=1)) 
    fphi = -fphi;

  return fphi;

}

void DTLocalTriggerTask::setQLabels(MonitorElement* me, short int iaxis){

  TH1* histo = me->getTH1();
  if (!histo) return;
  
  TAxis* axis=0;
  if (iaxis==1) {
    axis=histo->GetXaxis();
  }
  else if(iaxis==2) {
    axis=histo->GetYaxis();
  }
  if (!axis) return;

  char *labels[7] = {"LI","LO","HI","HO","LL","HL","HH"};
  int istart = axis->GetXmin()<-1 ? 2 : 1;
  for (int i=0;i<7;i++) {
    axis->SetBinLabel(i+istart,labels[i]);
  }

}

void DTLocalTriggerTask::loadDTTFMap(){

  ParameterSet emptyPS;
  ParameterSet mapParameters = parameters.getUntrackedParameter<ParameterSet>("DTTFMap",emptyPS);
  for (int dttf_wh=-2; dttf_wh<=2; ++dttf_wh){
    for (int dttf_sc=1; dttf_sc<=12; ++dttf_sc){
      stringstream parameter;
      parameter << "wh" << dttf_wh << "sec" << dttf_sc;
      vector<int> vec;
      vec.push_back(dttf_wh);
      vec.push_back(dttf_sc);
      vector<int> mapping = mapParameters.getUntrackedParameter<vector<int> >(parameter.str(),vec);
      int wh = mapping.at(0);
      int sc = mapping.at(1);
      mapDTTF[dttf_wh+2][dttf_sc][0] = wh;
      mapDTTF[dttf_wh+2][dttf_sc][1] = sc;
    }
  }

}

void DTLocalTriggerTask::correctMapping(int& wh, int& sector){

  int dttf_wh = wh;
  int dttf_sc = sector;
  wh = mapDTTF[dttf_wh+2][dttf_sc][0];
  sector = mapDTTF[dttf_wh+2][dttf_sc][1];
  

}




void DTLocalTriggerTask::triggerSource(const edm::Event& e) {
  
  
  if ( !parameters.getUntrackedParameter<bool>("localrun", true) ){
    
    Handle<LTCDigiCollection> ltcdigis;
    e.getByType(ltcdigis);
    
    for (std::vector<LTCDigi>::const_iterator ltc_it = ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
      
      int otherTriggerSum=0;
      for (int i = 1; i < 6; i++) {
	otherTriggerSum += int((*ltc_it).HasTriggered(i));
      }
      if ((*ltc_it).HasTriggered(0) && otherTriggerSum == 0) 
	trigsrc = "_DTonly";
      else if (!(*ltc_it).HasTriggered(0))
	trigsrc = "_NoDT";
      else if ((*ltc_it).HasTriggered(0) && otherTriggerSum > 0)
	trigsrc = "_DTalso";
      
    }
    return;
  }

  trigsrc = "";
  return;

}

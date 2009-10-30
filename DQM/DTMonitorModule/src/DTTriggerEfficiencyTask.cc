/*
 * \file DTTriggerEfficiencyTask.cc
 * 
 * $Date: 2009/07/29 10:30:30 $
 * $Revision: 1.1 $
 * \author C.Battilana - CIEMAT
 *
*/

#include "DQM/DTMonitorModule/src/DTTriggerEfficiencyTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// DT trigger
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

// Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// DT Digi
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>


//Root
#include"TH1.h"
#include"TAxis.h"

#include <sstream>
#include <iostream>
#include <fstream>


using namespace edm;
using namespace std;

DTTriggerEfficiencyTask::DTTriggerEfficiencyTask(const edm::ParameterSet& ps) : trigGeomUtils(0) {
  
  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask")  << "[DTTriggerEfficiencyTask]: Constructor" << endl;

  parameters = ps;
  dbe = edm::Service<DQMStore>().operator->();

}


DTTriggerEfficiencyTask::~DTTriggerEfficiencyTask() {

  edm::LogProblem ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask")  << "[DTTriggerEfficiencyTask]: analyzed " << nevents << " events" << endl;

}


void DTTriggerEfficiencyTask::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask") << "[DTTriggerEfficiencyTask]: BeginJob" << endl;

  detailedPlots = parameters.getUntrackedParameter<bool>("detailedAnalysis",false);
  processDCC = parameters.getUntrackedParameter<bool>("processDCC",true);    // CB Li metto untracked? Guarda resto del DQM...
  processDDU = parameters.getUntrackedParameter<bool>("processDDU",true);
  minBXDDU = parameters.getUntrackedParameter<int>("minBXDDU",0);
  maxBXDDU = parameters.getUntrackedParameter<int>("maxBXDDU",20);
  
  context.get<MuonGeometryRecord>().get(muonGeom);
  trigGeomUtils = new DTTrigGeomUtils(muonGeom);
  nevents = 0;


  for (int wh=-2;wh<=2;++wh){
    if (processDCC) {
      bookWheelHistos(wh,"DCC_TrigEffDenum");
      bookWheelHistos(wh,"DCC_TrigEffNum");
      bookWheelHistos(wh,"DCC_TrigEffCorrNum");
      if (detailedPlots) {
	for (int stat=1;stat<=4;++stat){
	  for (int sect=1;sect<=12;++sect){
	    DTChamberId dtChId(wh,stat,sect);   
	    bookChamberHistos(dtChId,"DCC_TrackPosvsAngleAnyQual","Segment");
	    bookChamberHistos(dtChId,"DCC_TrackPosvsAngleCorr","Segment");
	    bookChamberHistos(dtChId,"DCC_TrackPosvsAngle","Segment");
	  }
	}
      }
    }
    if (processDDU) {
      bookWheelHistos(wh,"DDU_TrigEffDenum");
      bookWheelHistos(wh,"DDU_TrigEffNum");
      bookWheelHistos(wh,"DDU_TrigEffCorrNum");
      if (detailedPlots) {
	for (int stat=1;stat<=4;++stat){
	  for (int sect=1;sect<=12;++sect){
	    DTChamberId dtChId(wh,stat,sect);   
	    bookChamberHistos(dtChId,"DDU_TrackPosvsAngleAnyQual","Segment");
	    bookChamberHistos(dtChId,"DDU_TrackPosvsAngleCorr","Segment");
	    bookChamberHistos(dtChId,"DDU_TrackPosvsAngle","Segment");
	  }
	}
      }
    }
  } // end of wheel loop

}

void DTTriggerEfficiencyTask::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask") <<"[DTTriggerEfficiencyTask]: Begin of LS transition"<<endl;
  
}


void DTTriggerEfficiencyTask::endJob(){

  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask")  << "[DTTriggerEfficiencyTask]: analyzed " << nevents << " events" << endl;
  if (processDDU) { dbe->rmdir(topFolder(0)); } // DDU top Folder
  if (processDCC) { dbe->rmdir(topFolder(1)); } // DCC top Folder

}


void DTTriggerEfficiencyTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;

  if (!hasRPCTriggers(e)) { return; }

  InputTag inputTagDCC = parameters.getUntrackedParameter<edm::InputTag>("inputTagDCC");
  InputTag inputTagDDU = parameters.getUntrackedParameter<edm::InputTag>("inputTagDDU");
  InputTag inputTagSEG = parameters.getUntrackedParameter<edm::InputTag>("inputTagSEG");

  for (int i=0;i<5;++i){
    for (int j=0;j<6;++j){
      for (int k=0;k<13;++k){
	phCodeBestDCC[j][i][k] = -1;
	phCodeBestDDU[j][i][k] = -1;
      }
    }
  }
  
  // Getting best DCC Stuff
  edm::Handle<L1MuDTChambPhContainer> l1DTTPGPh;
  e.getByLabel(inputTagDCC,l1DTTPGPh);
  vector<L1MuDTChambPhDigi>*  phTrigs = l1DTTPGPh->getContainer();
  
  vector<L1MuDTChambPhDigi>::const_iterator iph  = phTrigs->begin();
  vector<L1MuDTChambPhDigi>::const_iterator iphe = phTrigs->end();
  for(; iph !=iphe ; ++iph) {
    
    int phwheel = iph->whNum();
    int phsec   = iph->scNum() + 1; // DTTF numbering [0:11] -> DT numbering [1:12]
    int phst    = iph->stNum();
    int phcode  = iph->code();

    if(phcode>phCodeBestDCC[phwheel+3][phst][phsec] && phcode<7) {
      phCodeBestDCC[phwheel+3][phst][phsec]=phcode; 
      phBestDCC[phwheel+3][phst][phsec] = &(*iph);
    }
  }

  //Getting Best DDU Stuff
  Handle<DTLocalTriggerCollection> trigsDDU;
  e.getByLabel(inputTagDDU,trigsDDU);
  DTLocalTriggerCollection::DigiRangeIterator detUnitIt;

  for (detUnitIt=trigsDDU->begin();detUnitIt!=trigsDDU->end();++detUnitIt){
      
    const DTChamberId& id = (*detUnitIt).first;
    const DTLocalTriggerCollection::Range& range = (*detUnitIt).second;

    int wh = id.wheel();
    int sec = id.sector();
    int st = id.station();

    for (DTLocalTriggerCollection::const_iterator trigIt = range.first; trigIt!=range.second;++trigIt){
	
      int quality = trigIt->quality();

      if(quality>-1 && quality<7 &&
	 quality>phCodeBestDDU[wh+3][st][sec]) {
	phCodeBestDDU[wh+3][st][sec]=quality;
	phBestDDU[wh+3][st][sec] = &(*trigIt);
      }
    }
  }

  //Getting Best Segments
  vector<const DTRecSegment4D*> best4DSegments;

  Handle<DTRecSegment4DCollection> segments4D;
  e.getByLabel(inputTagSEG, segments4D);
  DTRecSegment4DCollection::const_iterator track;
  DTRecSegment4DCollection::id_iterator chamberId;

  for (chamberId = segments4D->id_begin(); chamberId != segments4D->id_end(); ++chamberId){
    
    DTRecSegment4DCollection::range  range = segments4D->get(*chamberId);
    const DTRecSegment4D* tmpBest=0;
    int tmpHit = 0;
    int hit = 0;

    for ( track = range.first; track != range.second; ++track){
      if( (*track).hasPhi() ) {
	hit = (*track).phiSegment()->degreesOfFreedom()+2;
	if ( hit>tmpHit ){
	  tmpBest = &(*track);
	  tmpHit = hit;
	  int sec = (*track).chamberId().sector();
	  if (sec==13){
	    sec=4;
	  }
	  else if (sec==14){
	    sec=10;
	  }
	}
      }
    }
    if (tmpBest) {
      best4DSegments.push_back(tmpBest);
    }
  }
    
  // Plot filling
  vector<const DTRecSegment4D*>::const_iterator btrack;
  for ( btrack = best4DSegments.begin(); btrack != best4DSegments.end(); ++btrack ){

    int wheel    = (*btrack)->chamberId().wheel();
    int station  = (*btrack)->chamberId().station();
    int scsector = 0;
    float x, xdir, y, ydir;
    trigGeomUtils->computeSCCoordinates((*btrack),scsector,x,xdir,y,ydir);
    int nHitsPhi = (*btrack)->phiSegment()->degreesOfFreedom()+2;	
    DTChamberId dtChId(wheel,station,scsector);
    uint32_t indexCh = dtChId.rawId(); 
    map<string, MonitorElement*> &innerChME = chamberHistos[indexCh];
    map<string, MonitorElement*> &innerWhME = wheelHistos[wheel];
    
    if (fabs(xdir)<30. && nHitsPhi>=7){
      
      if (processDCC) {
	int qual   = phCodeBestDCC[wheel+3][station][scsector];
	innerWhME.find("DCC_TrigEffDenum")->second->Fill(scsector,station);
	if ( qual>=0 && qual<7 ) {
	  innerWhME.find("DCC_TrigEffNum")->second->Fill(scsector,station);
	  if ( qual>=4 ) {
	    innerWhME.find("DCC_TrigEffCorrNum")->second->Fill(scsector,station);
	    // if (qual==7 ) {
// 	      innerWhME.find("DCC_TrackPosvsAngleHH")->second->Fill(scsector,wheel);
// 	    }
	  }
	}
	if (detailedPlots) {
	  innerChME.find("DCC_TrackPosvsAngle")->second->Fill(xdir,x);
	  if ( qual>=0 && qual<7 ) {
	    innerChME.find("DCC_TrackPosvsAngleAnyQual")->second->Fill(xdir,x);
	    if ( qual>=4 ) {
		innerChME.find("DCC_TrackPosvsAngleCorr")->second->Fill(xdir,x);	
		// if (qual==7 ) {
// 		  innerChME.find("DCC_TrackPosvsAngleHH")->second->Fill(xdir,x);
// 		}
	    }
	  }
	}
      }

      if (processDDU) {
	int qual   = phCodeBestDDU[wheel+3][station][scsector];
	innerWhME.find("DDU_TrigEffDenum")->second->Fill(scsector,station);
	bool qualOK = qual>=0 && qual<7;
	int bx = qualOK ? phBestDDU[wheel+3][station][scsector]->bx() : -10;
	if ( qualOK && bx>=minBXDDU && bx<=maxBXDDU ) {
	  innerWhME.find("DDU_TrigEffNum")->second->Fill(scsector,station);
	  if ( qual>=4 ) {
	    innerWhME.find("DDU_TrigEffCorrNum")->second->Fill(scsector,station);
	    // if (qual==7 ) {
// 	      innerWhME.find("DDU_TrackPosvsAngleHH")->second->Fill(scsector,wheel);
// 	    }
	  }
	}
	if (detailedPlots) {
	  innerChME.find("DDU_TrackPosvsAngle")->second->Fill(xdir,x);
	  if ( qualOK && bx>+minBXDDU && bx<maxBXDDU ) {
	    innerChME.find("DDU_TrackPosvsAngleAnyQual")->second->Fill(xdir,x);
	    if ( qual>=4 ) {
	      innerChME.find("DDU_TrackPosvsAngleCorr")->second->Fill(xdir,x);	
	      // if (qual==7 ) {
// 		innerChME.find("DDU_TrackPosvsAngleHH")->second->Fill(xdir,x);
// 	      }
	    }
	  }
	}
      }
    }
  }

}

bool DTTriggerEfficiencyTask::hasRPCTriggers(const edm::Event& e) {

  InputTag inputTagGMT = parameters.getUntrackedParameter<edm::InputTag>("inputTagGMT");
  edm::Handle<L1MuGMTReadoutCollection> gmtrc; 
  e.getByLabel(inputTagGMT,gmtrc);

  std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;
   
  for(igmtrr=gmt_records.begin(); igmtrr!=gmt_records.end(); igmtrr++) {
    if( (*igmtrr).getBxInEvent()==0 ) {
      std::vector<L1MuRegionalCand>::const_iterator iter1;
      std::vector<L1MuRegionalCand> rmc = (*igmtrr).getBrlRPCCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
	if ( !(*iter1).empty() ) {
	  return true;
	}
      }
    }
  }

  return false;
  
}

void DTTriggerEfficiencyTask::bookChamberHistos(const DTChamberId& dtCh, string histoTag, string folder) {
  
  int wh = dtCh.wheel();		
  int sc = dtCh.sector();	
  int st = dtCh.station();
  stringstream wheel; wheel << wh;	
  stringstream station; station << st;	
  stringstream sector; sector << sc;	

  string histoType     = histoTag.substr(4,histoTag.find("_",4)-4);
  string hwFolder      = topFolder(histoTag.substr(0,3)=="DCC"); 
  string bookingFolder = hwFolder + "Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" + station.str() + "/" + folder;
  string histoName     = histoTag + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  dbe->setCurrentFolder(bookingFolder);
    
  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask") << "[DTTriggerEfficiencyTask]: booking " << bookingFolder << "/" << histoName << endl;
    
  if( histoType.find("TrackPosvsAngle") == 0 ){
    float min, max;
    int nbins;
    trigGeomUtils->phiRange(dtCh,min,max,nbins,20);
    string histoLabel = "Position vs Angle (phi)";
    if (histoType.find("Corr")  != string::npos) histoLabel += " for correlated triggers";
    else if (histoType.find("AnyQual") != string::npos) histoLabel += " for any qual triggers";
    (chamberHistos[dtCh.rawId()])[histoTag] = dbe->book2D(histoName,histoLabel,12,-30.,30.,nbins,min,max);
    return ;
  }

}

void DTTriggerEfficiencyTask::bookWheelHistos(int wheel,string hTag,string folder) {
  
  stringstream wh; wh << wheel;
  string basedir;  
  bool isDCC = hTag.substr(0,3)=="DCC" ;  
  if (hTag.find("Summary") != string::npos ) {
    basedir = topFolder(isDCC);   //Book summary histo outside wheel directories
  } else {
    basedir = topFolder(isDCC) + "Wheel" + wh.str() + "/" ;
    
  }
  if (folder != "") {
    basedir += folder +"/" ;
  }
  dbe->setCurrentFolder(basedir);

  string hname    = hTag+ "_W" + wh.str();

  LogTrace("DTDQM|DTMonitorModule|DTTriggerEfficiencyTask") << "[DTTriggerEfficiencyTask]: booking "<< basedir << hname;
  
//   if (hTag.find("Phi")!= string::npos ||
//       hTag.find("Summary") != string::npos ){    
    MonitorElement* me = dbe->book2D(hname.c_str(),hname.c_str(),12,1,13,4,1,5);

    me->setBinLabel(1,"MB1",2);
    me->setBinLabel(2,"MB2",2);
    me->setBinLabel(3,"MB3",2);
    me->setBinLabel(4,"MB4",2);
    me->setAxisTitle("Sector",1);
    
    wheelHistos[wheel][hTag] = me;
    return;
    //   }
  
//   if (hTag.find("Theta") != string::npos){
//     MonitorElement* me =dbe->book2D(hname.c_str(),hname.c_str(),12,1,13,3,1,4);

//     me->setBinLabel(1,"MB1",2);
//     me->setBinLabel(2,"MB2",2);
//     me->setBinLabel(3,"MB3",2);
//     me->setAxisTitle("Sector",1);

//     wheelHistos[wheel][hTag] = me;
//     return;
//   }
  
}


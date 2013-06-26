/*
 * \file DTLocalTriggerBaseTask.cc
 * 
 * $Date: 2013/01/07 16:25:31 $
 * $Revision: 1.2 $
 * \author C. Battilana - CIEMAT
 *
*/

#include "DQM/DTMonitorModule/src/DTLocalTriggerBaseTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// DT DQM
#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"

// DT trigger
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

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

class DTTPGCompareUnit {
  
public:

  DTTPGCompareUnit()  { theQual[0]=-1 ; theQual[1]=-1; }
  ~DTTPGCompareUnit() { };

  void setDDU(int qual, int bx) { theQual[0] = qual; theBX[0] = bx; }
  void setDCC(int qual, int bx) { theQual[1] = qual; theBX[1] = bx; }

  bool hasOne()      const { return theQual[0]!=-1 || theQual[1]!=-1; };
  bool hasBoth()     const { return theQual[0]!=-1 && theQual[1]!=-1; };
  bool hasSameQual() const { return hasBoth() && theQual[0]==theQual[1]; };
  int  deltaBX() const { return theBX[0] - theBX[1]; }
  int  qualDDU() const { return theQual[0]; }
  int  qualDCC() const { return theQual[1]; }

private:
  
  int theQual[2]; // 0=DDU 1=DCC
  int theBX[2];   // 0=DDU 1=DCC

};    



DTLocalTriggerBaseTask::DTLocalTriggerBaseTask(const edm::ParameterSet& ps) : 
  nEvents(0), nLumis(0), theTrigGeomUtils(0) {
  
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
    << "[DTLocalTriggerBaseTask]: Constructor"<<endl;

  tpMode           = ps.getUntrackedParameter<bool>("testPulseMode");
  detailedAnalysis = ps.getUntrackedParameter<bool>("detailedAnalysis");

  targetBXDCC  = ps.getUntrackedParameter<int>("targetBXDCC");
  targetBXDDU  = ps.getUntrackedParameter<int>("targetBXDDU");
  bestAccRange = ps.getUntrackedParameter<int>("bestTrigAccRange");

  processDCC   = ps.getUntrackedParameter<bool>("processDCC");
  processDDU   = ps.getUntrackedParameter<bool>("processDDU");

  if (processDCC) theTypes.push_back("DCC");
  if (processDDU) theTypes.push_back("DDU");

  if (tpMode) {
    topFolder("DCC") = "DT/11-LocalTriggerTP-DCC/";
    topFolder("DDU") = "DT/12-LocalTriggerTP-DDU/";
  } else {
    topFolder("DCC") = "DT/03-LocalTrigger-DCC/";
    topFolder("DDU") = "DT/04-LocalTrigger-DDU/";
  }

  theParams = ps; 
  theDQMStore = edm::Service<DQMStore>().operator->();

}


DTLocalTriggerBaseTask::~DTLocalTriggerBaseTask() {

  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
    << "[DTLocalTriggerBaseTask]: analyzed " << nEvents << " events" << endl;
  if (theTrigGeomUtils) { delete theTrigGeomUtils; }

}


void DTLocalTriggerBaseTask::beginJob() {
 
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
    << "[DTLocalTriggerBaseTask]: BeginJob" << endl;

}


void DTLocalTriggerBaseTask::beginRun(const edm::Run& run, const edm::EventSetup& context) {

  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
    << "[DTLocalTriggerBaseTask]: BeginRun" << endl;   

  ESHandle<DTGeometry> theGeom;  
  context.get<MuonGeometryRecord>().get(theGeom);
  theTrigGeomUtils = new DTTrigGeomUtils(theGeom);

  theDQMStore->setCurrentFolder("DT/EventInfo/Counters");
  nEventMonitor = theDQMStore->bookFloat("nProcessedEventsTrigger");
  for (int wh=-2;wh<3;++wh){
    for (int stat=1;stat<5;++stat){
      for (int sect=1;sect<13;++sect){
	bookHistos(DTChamberId(wh,stat,sect));
      }
    }
    bookHistos(wh);
  }

}


void DTLocalTriggerBaseTask::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

  nEventsInLS=0;
  nLumis++;
  int resetCycle = theParams.getUntrackedParameter<int>("ResetCycle"); 

  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
    << "[DTLocalTriggerBaseTask]: Begin of LS transition" << endl;
  
  if( nLumis%resetCycle == 0 ) {
    map<uint32_t,map<string,MonitorElement*> >::const_iterator chambIt  = chamberHistos.begin();
    map<uint32_t,map<string,MonitorElement*> >::const_iterator chambEnd = chamberHistos.end();
    for(;chambIt!=chambEnd;++chambIt) {
      map<string,MonitorElement*>::const_iterator histoIt  = chambIt->second.begin();
      map<string,MonitorElement*>::const_iterator histoEnd = chambIt->second.end();
      for(;histoIt!=histoEnd;++histoIt) {
	histoIt->second->Reset();
      }
    }
  }
  
}

void DTLocalTriggerBaseTask::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
    << "[DTLocalTriggerBaseTask]: End of LS transition" << endl;
  
  map<uint32_t,DTTimeEvolutionHisto* >::const_iterator chambIt  = trendHistos.begin();
  map<uint32_t,DTTimeEvolutionHisto* >::const_iterator chambEnd = trendHistos.end();
  for(;chambIt!=chambEnd;++chambIt) {
    chambIt->second->updateTimeSlot(lumiSeg.luminosityBlock(), nEventsInLS);
  }
  
}


void DTLocalTriggerBaseTask::endJob() {

  LogVerbatim("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
    << "[DTLocalTriggerBaseTask]: analyzed " << nEvents << " events" << endl;

  if (processDCC) theDQMStore->rmdir(topFolder("DCC"));
  if (processDDU) theDQMStore->rmdir(topFolder("DDU"));

}


void DTLocalTriggerBaseTask::analyze(const edm::Event& e, const edm::EventSetup& c){
  
  nEvents++;
  nEventsInLS++;
  nEventMonitor->Fill(nEvents);

  theCompMap.clear();

  Handle<L1MuDTChambPhContainer> phiTrigsDCC;
  Handle<L1MuDTChambThContainer> thetaTrigsDCC;
  Handle<DTLocalTriggerCollection> trigsDDU;

  if (processDCC) {
    InputTag inputTagDCC = theParams.getUntrackedParameter<InputTag>("inputTagDCC");

    e.getByLabel(inputTagDCC,phiTrigsDCC);
    e.getByLabel(inputTagDCC,thetaTrigsDCC);

    if (phiTrigsDCC.isValid() && thetaTrigsDCC.isValid()) {
      runDCCAnalysis(phiTrigsDCC->getContainer(),thetaTrigsDCC->getContainer());
    } else {
      LogVerbatim("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
	<< "[DTLocalTriggerBaseTask]: one or more DCC handles for Input Tag " 
	<< inputTagDCC <<" not found!" << endl;
      return;
    }
  }

  if (processDDU) {
    InputTag inputTagDDU = theParams.getUntrackedParameter<InputTag>("inputTagDDU");
    e.getByLabel(inputTagDDU,trigsDDU);

    if (trigsDDU.isValid()) {
      runDDUAnalysis(trigsDDU);
    } else {
      LogVerbatim("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
	<< "[DTLocalTriggerBaseTask]: one or more DDU handles for Input Tag " 
	<< inputTagDDU <<" not found!" << endl;
      return;
    }
  }
  
  if (processDCC && processDDU)
    runDDUvsDCCAnalysis();

}


void DTLocalTriggerBaseTask::bookHistos(const DTChamberId& dtCh) {

  uint32_t rawId = dtCh.rawId();

  stringstream wheel; wheel << dtCh.wheel();	
  stringstream station; station << dtCh.station();	
  stringstream sector; sector << dtCh.sector();	

  map<string,int> minBX;
  map<string,int> maxBX;

  minBX["DCC"] = theParams.getUntrackedParameter<int>("minBXDCC");
  maxBX["DCC"] = theParams.getUntrackedParameter<int>("maxBXDCC");
  minBX["DDU"] = theParams.getUntrackedParameter<int>("minBXDDU");
  maxBX["DDU"] = theParams.getUntrackedParameter<int>("maxBXDDU");

  int nTimeBins  = theParams.getUntrackedParameter<int>("nTimeBins");
  int nLSTimeBin = theParams.getUntrackedParameter<int>("nLSTimeBin");

  string chTag = "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  vector<string>::const_iterator typeIt  = theTypes.begin();
  vector<string>::const_iterator typeEnd = theTypes.end();

  for (; typeIt!=typeEnd; ++typeIt) {

    LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
      << "[DTLocalTriggerBaseTask]: booking histos for " << topFolder((*typeIt)) << "Wheel" 
      << wheel.str() << "/Sector" << sector.str() << "/Station"<< station.str() << endl;

    // Book Phi View Related Plots
    theDQMStore->setCurrentFolder(topFolder(*typeIt) + "Wheel" + wheel.str() + "/Sector" 
			  + sector.str() + "/Station" + station.str() + "/LocalTriggerPhi");      

    string histoTag = (*typeIt) + "_BXvsQual";
    chamberHistos[rawId][histoTag] = theDQMStore->book2D(histoTag+chTag,"BX vs trigger quality",
       7,-0.5,6.5,(int)(maxBX[(*typeIt)]-minBX[*typeIt]+1),minBX[*typeIt]-.5,maxBX[*typeIt]+.5);
    setQLabels((chamberHistos[rawId])[histoTag],1);

    if (!tpMode) {
      histoTag = (*typeIt) + "_BestQual";
      chamberHistos[rawId][histoTag] = theDQMStore->book1D(histoTag+chTag,
	         "Trigger quality of best primitives",7,-0.5,6.5);
      setQLabels(chamberHistos[rawId][histoTag],1);

      histoTag = (*typeIt) + "_Flag1stvsQual";
      chamberHistos[dtCh.rawId()][histoTag] = theDQMStore->book2D(histoTag+chTag,
	          "1st/2nd trig flag vs quality",7,-0.5,6.5,2,-0.5,1.5);
      setQLabels(chamberHistos[rawId][histoTag],1);
    }

    if (*typeIt=="DCC") {
      float minPh, maxPh; int nBinsPh;
      theTrigGeomUtils->phiRange(dtCh,minPh,maxPh,nBinsPh);

      histoTag = (*typeIt) + "_QualvsPhirad";    
      chamberHistos[rawId][histoTag] = theDQMStore->book2D(histoTag+chTag,
           "Trigger quality vs local position",nBinsPh,minPh,maxPh,7,-0.5,6.5);
      setQLabels(chamberHistos[rawId][histoTag],2);

      if (detailedAnalysis && !tpMode) {
	histoTag = (*typeIt) + "_QualvsPhibend";
	chamberHistos[rawId][histoTag] = theDQMStore->book2D(histoTag+chTag,
	      "Trigger quality vs local direction",200,-40.,40.,7,-0.5,6.5);
	setQLabels((chamberHistos[dtCh.rawId()])[histoTag],2);
      }   
    }
          
    // Book Theta View Related Plots
    theDQMStore->setCurrentFolder(topFolder(*typeIt) + "Wheel" + wheel.str() + "/Sector" 
	         + sector.str() + "/Station" + station.str() + "/LocalTriggerTheta");

    if((*typeIt)=="DCC") {
      histoTag = (*typeIt) + "_PositionvsBX";      
      chamberHistos[rawId][histoTag] = theDQMStore->book2D(histoTag+chTag,"Theta trigger position vs BX",
			      (int)(maxBX[(*typeIt)]-minBX[*typeIt]+1),minBX[*typeIt]-.5,maxBX[*typeIt]+.5,7,-0.5,6.5);
    } else {
      histoTag = (*typeIt) + "_ThetaBXvsQual";      
      chamberHistos[rawId][histoTag] =  theDQMStore->book2D(histoTag+chTag,"BX vs trigger quality",7,-0.5,6.5,
					   (int)(maxBX[(*typeIt)]-minBX[*typeIt]+1),minBX[*typeIt]-.5,maxBX[*typeIt]+.5);
      setQLabels((chamberHistos[dtCh.rawId()])[histoTag],1);

      histoTag = (*typeIt) + "_ThetaBestQual";      
      chamberHistos[rawId][histoTag] = theDQMStore->book1D(histoTag+chTag,
      "Trigger quality of best primitives (theta)",7,-0.5,6.5);
      setQLabels((chamberHistos[dtCh.rawId()])[histoTag],1);
    }

  }

  if (processDCC && processDDU) {
    // Book DCC/DDU Comparison Plots
    theDQMStore->setCurrentFolder(topFolder("DDU") + "Wheel" + wheel.str() + "/Sector" 
		       + sector.str() + "/Station" + station.str() + "/LocalTriggerPhi");      

    string histoTag = "COM_QualDDUvsQualDCC";
    chamberHistos[rawId][histoTag] = theDQMStore->book2D(histoTag+chTag,
			"DDU quality vs DCC quality",8,-1.5,6.5,8,-1.5,6.5);
    setQLabels((chamberHistos[rawId])[histoTag],1);
    setQLabels((chamberHistos[rawId])[histoTag],2);

    histoTag = "COM_MatchingTrend";
    trendHistos[rawId] = new DTTimeEvolutionHisto(&(*theDQMStore),histoTag+chTag,
						  "Fraction of DDU-DCC matches w.r.t. proc evts",
						  nTimeBins,nLSTimeBin,true,0);
  }      

}

void DTLocalTriggerBaseTask::bookHistos(int wh) {
  
  stringstream wheel; wheel << wh;	
  theDQMStore->setCurrentFolder(topFolder("DDU") + "Wheel" + wheel.str() + "/");
  string whTag = "_W" + wheel.str();
    
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") 
    << "[DTLocalTriggerBaseTask]: booking wheel histos for " 
    << topFolder("DDU") << "Wheel" << wh << endl;
  
  string histoTag = "COM_BXDiff";
  MonitorElement *me = theDQMStore->bookProfile2D(histoTag+whTag,
        "DDU-DCC BX Difference",12,1,13,4,1,5,0.,20.);
  me->setAxisTitle("Sector",1);
  me->setAxisTitle("station",2);
  wheelHistos[wh][histoTag] = me;
  
}


void DTLocalTriggerBaseTask::runDCCAnalysis( std::vector<L1MuDTChambPhDigi>* phTrigs, 
					 std::vector<L1MuDTChambThDigi>* thTrigs ){

  vector<L1MuDTChambPhDigi>::const_iterator iph  = phTrigs->begin();
  vector<L1MuDTChambPhDigi>::const_iterator iphe = phTrigs->end();

  for(; iph !=iphe ; ++iph) {

    int wh    = iph->whNum();
    int sec   = iph->scNum() + 1; // DTTF->DT Convention
    int st    = iph->stNum();
    int qual  = iph->code();
    int is1st = iph->Ts2Tag() ? 1 : 0;
    int bx    = iph->bxNum() - is1st;

    if (qual <0 || qual>6) continue; // Check that quality is in a valid range

    DTChamberId dtChId(wh,st,sec);
    uint32_t rawId = dtChId.rawId();
      
    float pos = theTrigGeomUtils->trigPos(&(*iph));
    float dir = theTrigGeomUtils->trigDir(&(*iph));

    if (abs(bx-targetBXDCC)<= bestAccRange &&
	theCompMap[rawId].qualDCC() <= qual) 
      theCompMap[rawId].setDCC(qual,bx);
    
    map<string, MonitorElement*> &innerME = chamberHistos[rawId];
    if (tpMode) {
      innerME["DCC_BXvsQual"]->Fill(qual,bx);      // SM BX vs Qual Phi view (1st tracks) 
      innerME["DCC_QualvsPhirad"]->Fill(pos,qual); // SM Qual vs radial angle Phi view
    } else {
      innerME["DCC_BXvsQual"]->Fill(qual,bx);         // SM BX vs Qual Phi view (1st tracks) 
      innerME["DCC_Flag1stvsQual"]->Fill(qual,is1st); // SM Qual 1st/2nd track flag Phi view
      if (!is1st) innerME["DCC_QualvsPhirad"]->Fill(pos,qual);  // SM Qual vs radial angle Phi view ONLY for 1st tracks
      if (detailedAnalysis) {
	innerME["DCC_QualvsPhibend"]->Fill(dir,qual); // SM Qual vs bending Phi view
      }
    }
    
  } 

  vector<L1MuDTChambThDigi>::const_iterator ith  = thTrigs->begin();
  vector<L1MuDTChambThDigi>::const_iterator ithe = thTrigs->end();

  for(; ith != ithe; ++ith) {
    int wh  = ith->whNum();
    int sec = ith->scNum() + 1; // DTTF -> DT Convention
    int st  = ith->stNum();
    int bx  = ith->bxNum();
      
    int thcode[7];
    
    for (int pos=0; pos<7; pos++)
      thcode[pos] = ith->code(pos);    
      
    DTChamberId dtChId(wh,st,sec);
    uint32_t rawId = dtChId.rawId();   

    map<string, MonitorElement*> &innerME = chamberHistos[rawId];

    for (int pos=0; pos<7; pos++)
      if (thcode[pos])
	innerME["DCC_PositionvsBX"]->Fill(bx,pos); // SM BX vs Position Theta view

  }
  
  // Fill Quality plots with best DCC triggers (phi view)
  if (!tpMode) {
    map<uint32_t,DTTPGCompareUnit>::const_iterator compIt  = theCompMap.begin();
    map<uint32_t,DTTPGCompareUnit>::const_iterator compEnd = theCompMap.end();
    for (; compIt!=compEnd; ++compIt) {
      int bestQual = compIt->second.qualDCC();
      if (bestQual > -1) 
	chamberHistos[compIt->first]["DCC_BestQual"]->Fill(bestQual);  // SM Best Qual Trigger Phi view
    }
  }

}


void DTLocalTriggerBaseTask::runDDUAnalysis(Handle<DTLocalTriggerCollection>& trigsDDU){
    
  DTLocalTriggerCollection::DigiRangeIterator detUnitIt  = trigsDDU->begin();
  DTLocalTriggerCollection::DigiRangeIterator detUnitEnd = trigsDDU->end();

  for (; detUnitIt!=detUnitEnd; ++detUnitIt){
      
    const DTChamberId& chId = (*detUnitIt).first;
    uint32_t rawId = chId.rawId();

    const DTLocalTriggerCollection::Range& range = (*detUnitIt).second;
    DTLocalTriggerCollection::const_iterator trigIt = range.first;
    map<string, MonitorElement*> &innerME = chamberHistos[rawId];

    int bestQualTheta = -1;

    for (; trigIt!=range.second; ++trigIt){
	
      int qualPhi   = trigIt->quality();
      int qualTheta = trigIt->trTheta();
      int flag1st   = trigIt->secondTrack() ? 1 : 0;
      int bx = trigIt->bx();
      int bxPhi = bx - flag1st; // phi BX assign is different for 1st & 2nd tracks
      
      if( qualPhi>-1 && qualPhi<7 ) { // it is a phi trigger
	if (abs(bx-targetBXDDU) <= bestAccRange &&
	    theCompMap[rawId].qualDDU()<= qualPhi)
	  theCompMap[rawId].setDDU(qualPhi,bxPhi);
	if(tpMode) {	  
	  innerME["DDU_BXvsQual"]->Fill(qualPhi,bxPhi); // SM BX vs Qual Phi view	
	} else {
	  innerME["DDU_BXvsQual"]->Fill(qualPhi,bxPhi); // SM BX vs Qual Phi view	
	  innerME["DDU_Flag1stvsQual"]->Fill(qualPhi,flag1st); // SM Quality vs 1st/2nd track flag Phi view
	}
      }

      if( qualTheta>0 && !tpMode ){// it is a theta trigger & is not TP
	if (qualTheta > bestQualTheta){
	  bestQualTheta = qualTheta;
	}
	innerME["DDU_ThetaBXvsQual"]->Fill(qualTheta,bx); // SM BX vs Qual Theta view
      }
    }
    
    // Fill Quality plots with best ddu triggers
    if (!tpMode && theCompMap.find(rawId)!= theCompMap.end()) {
      int bestQualPhi = theCompMap[rawId].qualDDU();
      if (bestQualPhi>-1)
	innerME["DDU_BestQual"]->Fill(bestQualPhi); // SM Best Qual Trigger Phi view
      if(bestQualTheta>0) {
	innerME["DDU_ThetaBestQual"]->Fill(bestQualTheta); // SM Best Qual Trigger Theta view
      }  
    }
  }
  
}


void DTLocalTriggerBaseTask::runDDUvsDCCAnalysis(){

  map<uint32_t,DTTPGCompareUnit>::const_iterator compIt  = theCompMap.begin();
  map<uint32_t,DTTPGCompareUnit>::const_iterator compEnd = theCompMap.end();

  for (; compIt!=compEnd; ++compIt) {

    uint32_t rawId = compIt->first;
    DTChamberId chId(rawId);
    map<string, MonitorElement*> &innerME = chamberHistos[rawId];

    const DTTPGCompareUnit & compUnit = compIt->second;	  
    if ( compUnit.hasOne() ){
      innerME["COM_QualDDUvsQualDCC"]->Fill(compUnit.qualDCC(),compUnit.qualDDU());
    }
    if ( compUnit.hasBoth() ){
      wheelHistos[chId.wheel()]["COM_BXDiff"]->Fill(chId.sector(),chId.station(),compUnit.deltaBX());
      if (  compUnit.hasSameQual() ) {
	trendHistos[rawId]->accumulateValueTimeSlot(1);
      }
    }
  }

}


void DTLocalTriggerBaseTask::setQLabels(MonitorElement* me, short int iaxis){

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

  string labels[7] = {"LI","LO","HI","HO","LL","HL","HH"};
  int istart = axis->GetXmin()<-1 ? 2 : 1;
  for (int i=0;i<7;i++) {
    axis->SetBinLabel(i+istart,labels[i].c_str());
  }

}

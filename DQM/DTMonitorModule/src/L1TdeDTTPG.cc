/*
 * \file L1TdeDTTPG.cc
 * 
  * $Date: 2010/11/18 09:42:52 $
 * $Revision: 1.0 $
 * \author C. Battilana - CIEMAT
 * \author M. Meneghelli - INFN BO
 *
*/

#include "DQM/DTMonitorModule/src/L1TdeDTTPG.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Trig stuff
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

//Root
#include"TH1.h"
#include"TH1F.h"
#include"TAxis.h"

//C++
#include <sstream>
#include <iostream>
#include <math.h>

using namespace edm;
using namespace std;

L1TdeDTTPG::L1TdeDTTPG(const ParameterSet& parameters) {
  
  LogTrace("L1TdeDTTPG") << "[L1TdeDTTPG]: Constructor"<<endl;

  theBaseFolder  = "L1TEMU/DTTPGexpert";
  theParams      = parameters;
  theDQMStore    = Service<DQMStore>().operator->();
  
  theDataTag = parameters.getUntrackedParameter<InputTag>("dataTag");
  theEmuTag  = parameters.getUntrackedParameter<InputTag>("emulatorTag");
  theGmtTag  = parameters.getUntrackedParameter<InputTag>("gmtTag");
  
  theDetailedAnalysis = parameters.getUntrackedParameter<bool>("detailedAnalysis");

}


L1TdeDTTPG::~L1TdeDTTPG() {

  LogTrace("L1TdeDTTPG") << "[L1TdeDTTPG]: Destructor"<< endl;

}


void L1TdeDTTPG::beginJob(){
 
  LogTrace("L1TdeDTTPG") << "[L1TdeDTTPG]: BeginJob" << endl;
  theEvents = 0;
  theLumis  = 0;

}


void L1TdeDTTPG::beginRun(const Run& run, const EventSetup& context) {

  LogTrace("L1TdeDTTPG") << "[L1TdeDTTPG]: BeginRun" << endl;   

  for (int wh=-2;wh<=2;++wh){
    for (int sect=1;sect<=12;++sect){
      for (int stat=1;stat<=4;++stat){
	bookHistos(DTChamberId(wh,stat,sect));
      }
    }
  }
  bookBarrelHistos();

  ESHandle<DTConfigManager> dttpgConf;
  context.get<DTConfigManagerRcd>().get(dttpgConf);
  barrelHistos["hEventSetupStatus"]->Fill(dttpgConf->CCBConfigValidity());
		
}


void L1TdeDTTPG::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  
  LogTrace("L1TdeDTTPG") << "[L1TdeDTTPG]: Begin of LS transition" << endl;
  theLumis++;

  if(theLumis%theParams.getUntrackedParameter<int>("ResetCycle") == 0) {
    map<uint32_t, map<string, MonitorElement*> > ::const_iterator chIt  = chHistos.begin();
    map<uint32_t, map<string, MonitorElement*> > ::const_iterator chEnd = chHistos.end();
    for(; chIt!=chEnd; ++chIt) {
      map<string, MonitorElement*> ::const_iterator histoIt  = chIt->second.begin();
      map<string, MonitorElement*> ::const_iterator histoEnd = chIt->second.end();
      for(; histoIt!=histoEnd; ++histoIt) {
	(*histoIt).second->Reset();
      }
    }
  }
  
}

void L1TdeDTTPG::endLuminosityBlock(const LuminosityBlock&  lumiSeg, const  EventSetup& context){

}


void L1TdeDTTPG::endJob(){

  LogTrace("L1TdeDTTPG") << "[L1TdeDTTPG]: analyzed " << theEvents << " events" << endl;
  theDQMStore->rmdir(topFolder());

}


void L1TdeDTTPG::analyze(const Event& event, const EventSetup& context){  

  bool hasGmtCand = false;
  Handle<L1MuGMTReadoutCollection> gmtrc; 
  event.getByLabel(theGmtTag,gmtrc);

  vector<L1MuRegionalCand> cands = gmtrc->getRecord(0).getDTBXCands();
  vector<L1MuRegionalCand>::const_iterator candIt  = cands.begin();
  vector<L1MuRegionalCand>::const_iterator candEnd = cands.end();
      
  for(; candIt!=candEnd; ++candIt) {
    if (!candIt->empty()) {
      hasGmtCand = true;
      break;
    }
  }

  if (!hasGmtCand) return; // do not perform comparisons in case of no GMT DT cands

  theEvents++;
  theCompareUnits.clear();

  barrelHistos["nProcessedEvts"]->Fill(theEvents);  

  Handle<L1MuDTChambPhContainer> data;
  event.getByLabel(theDataTag,data);

  Handle<L1MuDTChambPhContainer> emulator;
  event.getByLabel(theEmuTag,emulator);

  fillCompareUnits(data,false);
  fillCompareUnits(emulator,true);

  map<int, DTTPGCompareUnit>::const_iterator compUnitIt  = theCompareUnits.begin();
  map<int, DTTPGCompareUnit>::const_iterator compUnitEnd = theCompareUnits.end();

  for(; compUnitIt!=compUnitEnd; ++compUnitIt) {

    uint32_t rawId = compUnitIt->second.getChId().rawId();
    if (compUnitIt->second.hasBoth()) {
      chHistos[rawId]["hDeltaQuality"]->Fill(compUnitIt->second.deltaQual());
      chHistos[rawId]["hDeltaQualityVsPhi"]->Fill(compUnitIt->second.phi(0),compUnitIt->second.deltaQual());
      if (compUnitIt->second.hasBothCorr()) {
	chHistos[rawId]["hDeltaPhi"]->Fill(max(min(double(compUnitIt->second.deltaPhi()),9.9),-9.9));
	if (compUnitIt->second.getChId().station() != 3 && compUnitIt->second.hasSameQual()) { // has same qual and is correlated
	  chHistos[rawId]["hDeltaPhiBend"]->Fill(max(min(double(compUnitIt->second.deltaPhiB()),9.9),-9.9));
	}
      }
      chHistos[rawId]["hEntries"]->Fill(0);
    } else {
      bool isEmu   = compUnitIt->second.hasEmu();
      bool isSecond = compUnitIt->second.getSecondBit(isEmu);
      int entry =  isEmu ? isSecond ? 4 : 2 : isSecond ? 3 : 1;
      if (!isEmu && isSecond) 
	chHistos[rawId]["hData2ndPhi"]->Fill(compUnitIt->second.phi(0));
      chHistos[rawId]["hEntries"]->Fill(entry);
    }
    
  }
  
}


void L1TdeDTTPG::fillCompareUnits(Handle<L1MuDTChambPhContainer> primHandle, bool isEmu) {

  vector<L1MuDTChambPhDigi> *primitives = primHandle->getContainer();

  vector<L1MuDTChambPhDigi>::const_iterator primIt  = primitives->begin();
  vector<L1MuDTChambPhDigi>::const_iterator primEnd = primitives->end();

  for(; primIt!=primEnd ; ++primIt) {

    int second  = primIt->Ts2Tag() ? 1 : 0;
    int bx      = primIt->bxNum() - second;

    if (theDetailedAnalysis)
      barrelHistos[isEmu ? "hBXEmuPrims" : "hBXDataPrims"]->Fill(bx);

    if ( abs(bx)==0 ) {

      int wh      = primIt->whNum();
      int sec     = primIt->scNum() + 1; // DTTF -> DT sector numbering
      int st      = primIt->stNum();
      int qual    = primIt->code();
      int phi     = primIt->phi();
      int phib    = primIt->phiB();
    

      int tpgId = tpgRawId(wh,sec,st,bx,second);
      theCompareUnits[tpgId].setQual(qual,isEmu);
      theCompareUnits[tpgId].setPhi(phi,isEmu);
      theCompareUnits[tpgId].setPhiB(phib,isEmu);
      theCompareUnits[tpgId].setSecondBit(second,isEmu);
      theCompareUnits[tpgId].setChId(DTChamberId(wh,st,sec));

      chHistos[DTChamberId(wh,st,sec).rawId()][isEmu ? "hEmuQuality" : "hDataQuality"]->Fill(qual);
      
    }

  }

  if (theDetailedAnalysis) 
    barrelHistos[isEmu ? "hNEmuPrims" : "hNDataPrims"]->Fill(primitives->size());

}

void L1TdeDTTPG::bookHistos(const DTChamberId& chId) {
  
  uint32_t rawId = chId.rawId();

  stringstream wheel; wheel << chId.wheel();	
  stringstream station; station << chId.station();	
  stringstream sector; sector << chId.sector();	

  string path = topFolder() + "/Wheel" + wheel.str() 
    + "/Sector" + sector.str() + "/Station" + station.str() ;
  
  theDQMStore->setCurrentFolder(path);
  
  LogTrace("L1TdeDTTPG") << "[L1TdeDTTPG]: booking histos in : " 
			 << path << endl;

  string chTag = "_W" + wheel.str() + "_Sec" 
     + sector.str() + "_St" + station.str();
  
  chHistos[rawId]["hDeltaQuality"] = 
    theDQMStore->book1D("hDeltaQuality"+chTag,"Data - Emu quality difference",13,-6.5,6.5);
  chHistos[rawId]["hDeltaQualityVsPhi"] = 
    theDQMStore->book2D("hDeltaQualityVsPhi"+chTag,"Data - Emu quality difference vs data phi coord",64,-2048,2048,13,-6.5,6.5);
  chHistos[rawId]["hDataQuality"] = 
    theDQMStore->book1D("hDataQuality"+chTag,"Data quality",7,-.5,6.5);
  chHistos[rawId]["hData2ndPhi"] = 
    theDQMStore->book1D("hData2ndPhi"+chTag,"Phi distribution of data only 2nd tracks",64,-2048,2048);
  chHistos[rawId]["hEmuQuality"] = 
    theDQMStore->book1D("hEmuQuality"+chTag,"Emulator quality",7,-.5,6.5);
  chHistos[rawId]["hDeltaPhi"] = 
    theDQMStore->book1D("hDeltaPhi"+chTag,"Data - Emu phi assignement difference",21,-10.5,10.5);
  if (chId.station() != 3) {
    chHistos[rawId]["hDeltaPhiBend"] = 
      theDQMStore->book1D("hDeltaPhiBend"+chTag,"Data - Emu phi bending assignement  difference",21,-10.5,10.5);
  }

  MonitorElement *hEntries = theDQMStore->book1D("hEntries"+chTag,"Occupancy",5,-.5,4.5);
  hEntries->setBinLabel(1,"Data & Emu",1);
  hEntries->setBinLabel(2,"Data Only 1st",1);
  hEntries->setBinLabel(3,"Emu Only 1st",1);
  hEntries->setBinLabel(4,"Data Only 2nd",1);
  hEntries->setBinLabel(5,"Emu Only 2nd",1);
  chHistos[rawId]["hEntries"] = hEntries;

}


void L1TdeDTTPG::bookBarrelHistos() {

  theDQMStore->setCurrentFolder(topFolder());
  
  LogTrace("L1TdeDTTPG") << "[L1TdeDTTPG]: booking histos in : " 
			 << topFolder() << endl;  
  
  if (theDetailedAnalysis) {
    barrelHistos["hNEmuPrims"]   = 
      theDQMStore->book1D("hNEmuPrims","Number of Emulated Primitives",20,-0.5,19.5);
    barrelHistos["hNDataPrims"]  = 
      theDQMStore->book1D("hNDataPrims","Number of HW Primitives",20,-0.5,19.5);
    barrelHistos["hBXEmuPrims"]  = 
      theDQMStore->book1D("hBXEmuPrims","BX of Emulated Primitives",11,-5.5,5.5);
    barrelHistos["hBXDataPrims"] = 
      theDQMStore->book1D("hBXDataPrims","BX of Real Primitives",11,-5.5,5.5);
  }

  barrelHistos["nProcessedEvts"] = theDQMStore->bookFloat("nProcessedEvts"); 
  barrelHistos["hEventSetupStatus"] = 
    theDQMStore->book1D("hEventSetupStatus","Configuration Status",11,-0.5,10.5);

}

int L1TdeDTTPG::tpgRawId(int wh, int st, int sec, int bx, int second) {

  int tpgId = second + 2*(bx+2) + 12*(sec-1) + 144*(st-1) + 576*(wh+2); 

  return tpgId;

}


uint32_t L1TdeDTTPG::tpgIdToChId(int tpgId) {

  uint32_t rawId =DTChamberId(tpgId/480-2,tpgId%480/120,tpgId%120%10).rawId();

  return rawId;

}

DTTPGCompareUnit::DTTPGCompareUnit() { 

  theQual[0] = 7;
  theQual[1] = 7; 

}


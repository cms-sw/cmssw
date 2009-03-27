/*  \author Anna Cimmino*/
#include <sstream>

#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DQM Services
#include "DQMServices/Core/interface/DQMStore.h"

using namespace edm;
using namespace std;
RPCEventSummary::RPCEventSummary(const ParameterSet& ps ){
  LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Constructor";

  //
  numberDisk_=3;

  enableReportSummary_ = ps.getUntrackedParameter<bool>("EnableSummaryReport",true);
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 1);
  eventInfoPath_ = ps.getUntrackedParameter<string>("EventInfoPath", "RPC/EventInfo");
  summaryFolder_ = ps.getUntrackedParameter<string>("RPCSummaryFolder", "RPC/RecHits/SummaryHistograms");
  verbose_=ps.getUntrackedParameter<bool>("VerboseLevel", 0);
 
  tier0_=ps.getUntrackedParameter<bool>("Tier0", false);

}

RPCEventSummary::~RPCEventSummary(){
  LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Destructor ";
  dbe_=0;
}

void RPCEventSummary::beginJob(const EventSetup& iSetup){
 LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin job ";
 dbe_ = Service<DQMStore>().operator->();
 dbe_->setVerbose(verbose_);
}

void RPCEventSummary::beginRun(const Run& r, const EventSetup& c){
 LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin run";

 MonitorElement* me;
 dbe_->setCurrentFolder(eventInfoPath_);

 //a global summary float [0,1] providing a global summary of the status 
 //and showing the goodness of the data taken by the the sub-system 
 string histoName="reportSummary";
 if ( me = dbe_->get(eventInfoPath_ +"/"+ histoName) ) {
    dbe_->removeElement(me->getName());
  }

  me = dbe_->bookFloat(histoName);
  me->Fill(1);

  //TH2F ME providing a mapof values[0-1] to show if problems are localized or distributed
  if ( me = dbe_->get(eventInfoPath_ + "/reportSummaryMap") ) {
     dbe_->removeElement(me->getName());
  }
  me = dbe_->book2D("reportSummaryMap", "RPC Report Summary Map", 15, -7.5, 7.5, 12, 0.5 ,12.5);
   cout<<__LINE__<<endl;
  //customize the 2d histo
  stringstream BinLabel;
  for (int i= 1 ; i<=15; i++){
    BinLabel.str("");
    if(i<13){
      BinLabel<<"Sec"<<i;
       me->setBinLabel(i,BinLabel.str(),2);
    } 

    BinLabel.str("");
    if(i<5)
      BinLabel<<"Disk"<<i-5;
    else if(i>11)
      BinLabel<<"Disk"<<i-11;
    else if(i==11 || i==5)
      BinLabel.str("");
    else
      BinLabel<<"Wheel"<<i-8;
 
     me->setBinLabel(i,BinLabel.str(),1);
  }

  //fill the histo with "1" --- just for the moment
  for(int i=1; i<=15; i++){
     for (int j=1; j<=12; j++ ){
       if(i==5 || i==11 || (j>6 && (i<6 || i>10)))    
	 me->setBinContent(i,j,-1);//bins that not correspond to subdetector parts
       else
	 me->setBinContent(i,j,1);
     }
   }

  if(numberDisk_ < 4)
    for (int j=1; j<=12; j++ ){
	me->setBinContent(1,j,-1);//bins that not correspond to subdetector parts
	me->setBinContent(15,j,-1);
    }

 //the reportSummaryContents folder containins a collection of ME floats [0-1] (order of 5-10)
 // which describe the behavior of the respective subsystem sub-components.
  dbe_->setCurrentFolder(eventInfoPath_+ "/reportSummaryContents");
  
  stringstream segName;
  vector<string> segmentNames;
  for(int i=-4; i<=4; i++){
    if(i>-3 && i<3) {
      segName.str("");
      segName<<"RPC_Wheel"<<i;
      segmentNames.push_back(segName.str());
    }
    if(i==0) continue;
    segName.str("");
    segName<<"RPC_Disk"<<i;
    segmentNames.push_back(segName.str());
  }
  

  for(unsigned int i=0; i<segmentNames.size(); i++){
    if ( me = dbe_->get(eventInfoPath_ + "/reportSummaryContents/" +segmentNames[i]) ) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->bookFloat(segmentNames[i]);
    me->Fill(1);
  }

  //excluded endcap parts
  if(numberDisk_ < 4){
    me=dbe_->get(eventInfoPath_ + "/reportSummaryContents/RPC_Disk4");
    if(me)  me->Fill(-1);
    me=dbe_->get(eventInfoPath_ + "/reportSummaryContents/RPC_Disk-4");
    if(me)  me->Fill(-1);
  }
}

void RPCEventSummary::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCEventSummary::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCEventSummary::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCEventSummary]: End of LS transition, performing DQM client operation";

  // counts number of lumiSegs 
   nLumiSegs_ = lumiSeg.id().luminosityBlock();
   stringstream meName;

  //check some statements and prescale Factor
  if(!enableReportSummary_  ||  (nLumiSegs_%prescaleFactor_ != 0)) return;

  MonitorElement * myMe;

  meName.str("");
  meName<<eventInfoPath_ + "/reportSummaryMap";
  MonitorElement * reportMe = dbe_->get(meName.str());
  
  MonitorElement * globalMe;

  //BARREL
  float barrelFactor =0;
  for(int w = -2 ; w<3; w++){
 
    meName.str("");
    meName<<summaryFolder_<<"/RPCChamberQuality_Roll_vs_Sector_Wheel"<<w;
    myMe = dbe_->get(meName.str());
  
    if(myMe){      
      float wheelFactor = 0;

      for(int s = 1; s<=myMe->getNbinsX() ; s++){
	float sectorFactor = 0;
	int rollInSector = 0;

     
	for(int r = 1;r<=myMe->getNbinsY(); r++){
	  if((s!=4 && r > 17 ) || ((s ==9 ||s ==10)  && r >15 ) )  continue;
	  rollInSector++;


	  if(myMe->getBinContent(s,r) == PARTIALLY_DEAD) sectorFactor+=0.8;
	  else if(myMe->getBinContent(s,r) == DEAD )sectorFactor+=0;
	  else sectorFactor+=1;	

	}
	if(rollInSector!=0)
	  sectorFactor = sectorFactor/rollInSector;

	 if(reportMe)	reportMe->setBinContent(w+8, s, sectorFactor);
	wheelFactor += sectorFactor;

      }//end loop on sectors

      wheelFactor = wheelFactor/myMe->getNbinsX();

      meName.str("");
      meName<<eventInfoPath_ + "/reportSummaryContents/RPC_Wheel"<<w; 
      globalMe=dbe_->get(meName.str());
      if(globalMe) globalMe->Fill(wheelFactor);

      barrelFactor += wheelFactor;
    }//

  
  }//end loop on wheel


  barrelFactor=barrelFactor/5;


  //ENDCAPS



  //Fill repor summary
  globalMe = dbe_->get(eventInfoPath_ +"/reportSummary"); 
  if(globalMe) globalMe->Fill(barrelFactor);


}

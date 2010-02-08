/*  \author Anna Cimmino*/
#include <sstream>

#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>
//CondFormats
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
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

  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 3);

  enableReportSummary_ = ps.getUntrackedParameter<bool>("EnableSummaryReport",true);
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 1);
  eventInfoPath_ = ps.getUntrackedParameter<string>("EventInfoPath", "RPC/EventInfo");
  globalFolder_ = ps.getUntrackedParameter<string>("RPCSummaryFolder", "RPC/RecHits/SummaryHistograms");
  minimumEvents_= ps.getUntrackedParameter<int>("MinimumRPCEvents", 10000);
 
  tier0_=ps.getUntrackedParameter<bool>("Tier0", false);


  FEDRange_.first  = ps.getUntrackedParameter<unsigned int>("MinimumRPCFEDId", 790);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumRPCFEDId", 792);
  
  NumberOfFeds_ =FEDRange_.second -  FEDRange_.first +1;
}

RPCEventSummary::~RPCEventSummary(){
  LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Destructor ";
  dbe_=0;
}

void RPCEventSummary::beginJob(const EventSetup& iSetup){
 LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin job ";
 dbe_ = Service<DQMStore>().operator->();
}

void RPCEventSummary::beginRun(const Run& r, const EventSetup& c){
 LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin run";

 edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

 int defaultValue = 1;

  if(0 != c.find( recordKey ) ) {
    defaultValue = -1;

    //get fed summary information
    ESHandle<RunInfo> sumFED;
    c.get<RunInfoRcd>().get(sumFED);    
    vector<int> FedsInIds= sumFED->m_fed_in;   
    unsigned  int f = 0;
    bool flag = false;
    while(!flag && f < FedsInIds.size()) {
      int fedID=FedsInIds[f];
      //make sure fed id is in allowed range  
      if(fedID>=FEDRange_.first && fedID<=FEDRange_.second) {
	defaultValue = 1;
	flag = true;
      } 
      f++;
    }   
  }   


 init_=false;

 MonitorElement* me;
 dbe_->setCurrentFolder(eventInfoPath_);

 //a global summary float [0,1] providing a global summary of the status 
 //and showing the goodness of the data taken by the the sub-system 
 string histoName="reportSummary";
 me =0;
 me = dbe_->get(eventInfoPath_ +"/"+ histoName);
 if ( 0!=me) {
    dbe_->removeElement(me->getName());
  }

  me = dbe_->bookFloat(histoName);
  me->Fill(defaultValue);

  //TH2F ME providing a mapof values[0-1] to show if problems are localized or distributed
  me =0;
  me = dbe_->get(eventInfoPath_ +"/reportSummaryMap");
  if ( 0!=me) {
    dbe_->removeElement(me->getName());
  }
  me = dbe_->book2D("reportSummaryMap", "RPC Report Summary Map", 15, -7.5, 7.5, 12, 0.5 ,12.5);
 
  //customize the 2d histo
  stringstream BinLabel;
  for (int i= 1 ; i<13; i++){//label sectors
    BinLabel.str("");
    BinLabel<<"Sec"<<i;
    me->setBinLabel(i,BinLabel.str(),2);
  }

  for (int w = -2 ; w<= 2; w++){//label wheels
    BinLabel.str("");
    BinLabel<<"Wheel"<<w;
    me->setBinLabel((w+8),BinLabel.str(),1);
  }

  for(int d = 1; d <= numberOfDisks_; d++ ){//label disks
    BinLabel.str("");
    BinLabel<<"Disk"<<d;
    me->setBinLabel((d+11),BinLabel.str(),1);
    BinLabel.str("");
    BinLabel<<"Disk"<<-d;
    me->setBinLabel((-d+5),BinLabel.str(),1);
  }


  //fill the histo with "1" --- just for the moment
  for(int i=1; i<=15; i++){
     for (int j=1; j<=12; j++ ){
       if(i==5 || i==11 || (j>6 && (i<6 || i>10)))    
	 me->setBinContent(i,j,-1);//bins that not correspond to subdetector parts
       else
	 me->setBinContent(i,j,defaultValue);
     }
   }

  if(numberOfDisks_ < 4){
    for (int j=1; j<=12; j++ ){
	me->setBinContent(1,j,-1);//bins that not correspond to subdetector parts
	me->setBinContent(15,j,-1);
    }
 }

 //the reportSummaryContents folder containins a collection of ME floats [0-1] (order of 5-10)
 // which describe the behavior of the respective subsystem sub-components.
  dbe_->setCurrentFolder(eventInfoPath_+ "/reportSummaryContents");
  
  stringstream segName;
  vector<string> segmentNames;
  for(int i=-2; i<=2; i++){//Wheels
    segName.str("");
    segName<<"RPC_Wheel"<<i;
    segmentNames.push_back(segName.str());
  }


  for(int i=1; i<=numberOfDisks_; i++){//Disks
    segName.str("");
    segName<<"RPC_Disk"<<i;
    segmentNames.push_back(segName.str());
    segName.str("");
    segName<<"RPC_Disk"<<-i;
    segmentNames.push_back(segName.str());
  }
  

  for(unsigned int i=0; i<segmentNames.size(); i++){
    me =0;
    me = dbe_->get(eventInfoPath_ + "/reportSummaryContents/" +segmentNames[i]);
    if ( 0!=me) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->bookFloat(segmentNames[i]);
    me->Fill(defaultValue);
  }

}

void RPCEventSummary::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCEventSummary::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCEventSummary::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCEventSummary]: End of LS transition, performing DQM client operation";

   
  MonitorElement * RPCEvents = dbe_->get(globalFolder_ +"/RPCEvents");  
  float   rpcevents = RPCEvents -> getEntries();

   if(!init_ && rpcevents < minimumEvents_) return;
   else if(!init_) {
     init_=true;
     nLumiSegs_ = prescaleFactor_;
   }else nLumiSegs_++;
   
   if (nLumiSegs_ % prescaleFactor_ != 0 ) return;
   
    stringstream meName;
   MonitorElement * myMe;
   
   meName.str("");
   meName<<eventInfoPath_ + "/reportSummaryMap";
   MonitorElement * reportMe = dbe_->get(meName.str());
   
   MonitorElement * globalMe;
   
   //BARREL
   float barrelFactor =0;
   for(int w = -2 ; w<3; w++){
     
     meName.str("");
     meName<<globalFolder_<<"/RPCChamberQuality_Roll_vs_Sector_Wheel"<<w;
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

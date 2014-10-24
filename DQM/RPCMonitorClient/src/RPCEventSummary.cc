/*  \author Anna Cimmino*/
#include <sstream>

#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>
//CondFormats
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"


RPCEventSummary::RPCEventSummary(const edm::ParameterSet& ps ){
  edm::LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Constructor";

  enableReportSummary_ = ps.getUntrackedParameter<bool>("EnableSummaryReport",true);
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 1);
  eventInfoPath_ = ps.getUntrackedParameter<std::string>("EventInfoPath", "RPC/EventInfo");


  std::string subsystemFolder = ps.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  std::string recHitTypeFolder = ps.getUntrackedParameter<std::string>("RecHitTypeFolder", "AllHits");
  std::string summaryFolder = ps.getUntrackedParameter<std::string>("SummaryFolder", "SummaryHistograms");

  globalFolder_  =  subsystemFolder +"/"+  recHitTypeFolder +"/"+ summaryFolder ;
  prefixFolder_  =  subsystemFolder +"/"+  recHitTypeFolder ;

  minimumEvents_= ps.getUntrackedParameter<int>("MinimumRPCEvents", 10000);
  numberDisk_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  doEndcapCertification_ = ps.getUntrackedParameter<bool>("EnableEndcapSummary", false);

  FEDRange_.first  = ps.getUntrackedParameter<unsigned int>("MinimumRPCFEDId", 790);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumRPCFEDId", 792);
  
  NumberOfFeds_ =FEDRange_.second -  FEDRange_.first +1;

  offlineDQM_ = ps.getUntrackedParameter<bool> ("OfflineDQM",true); 


}

RPCEventSummary::~RPCEventSummary(){
  edm::LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Destructor ";
  dbe_=0;
}

void RPCEventSummary::beginJob(){
 edm::LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin job ";
 dbe_ = edm::Service<DQMStore>().operator->();
}

void RPCEventSummary::beginRun(const edm::Run& r, const edm::EventSetup& setup){
 edm::LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin run";
  
  init_ = false;  
  lumiCounter_ = prescaleFactor_ ;

 edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
 
 int defaultValue = 1;
 
 if(0 != setup.find( recordKey ) ) {
   defaultValue = -1;
   //get fed summary information
   edm::ESHandle<RunInfo> sumFED;
   setup.get<RunInfoRcd>().get(sumFED);    
   std::vector<int> FedsInIds= sumFED->m_fed_in;   
   unsigned int f = 0;
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
 

 MonitorElement* me;
 dbe_->setCurrentFolder(eventInfoPath_);

 //a global summary float [0,1] providing a global summary of the status 
 //and showing the goodness of the data taken by the the sub-system 
 std::string histoName="reportSummary";
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
  std::stringstream BinLabel;
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
	 me->setBinContent(i,j,defaultValue);
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
  
  std::stringstream segName;
  std::vector<std::string> segmentNames;
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
    me =0;
    me = dbe_->get(eventInfoPath_ + "/reportSummaryContents/" +segmentNames[i]);
    if ( 0!=me) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->bookFloat(segmentNames[i]);
    me->Fill(defaultValue);
  }

  //excluded endcap parts
  if(numberDisk_ < 4){
    me=dbe_->get(eventInfoPath_ + "/reportSummaryContents/RPC_Disk4");
    if(me)  me->Fill(-1);
    me=dbe_->get(eventInfoPath_ + "/reportSummaryContents/RPC_Disk-4");
    if(me)  me->Fill(-1);
  }
}

void RPCEventSummary::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context){} 

void RPCEventSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {}

void RPCEventSummary::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {  
  edm::LogVerbatim ("rpceventsummary") <<"[RPCEventSummary]: End of LS transition, performing DQM client operation";

  if(offlineDQM_) return;

  if(!init_){
    this->clientOperation();
    return;
  }

  lumiCounter_++;
  if(lumiCounter_%prescaleFactor_ != 0) return;

  this->clientOperation();

}

void RPCEventSummary::endRun(const edm::Run& r, const edm::EventSetup& c){
  
  this->clientOperation();
}

void RPCEventSummary::clientOperation(){

  float  rpcevents = minimumEvents_;
  RPCEvents = dbe_->get( prefixFolder_  +"/RPCEvents");  

  if(RPCEvents) {
    rpcevents = RPCEvents ->getBinContent(1);
  }
  

  if(rpcevents < minimumEvents_) return;
  init_ = true;
  std::stringstream meName;
  MonitorElement * myMe;
   
  meName.str("");
  meName<<eventInfoPath_ + "/reportSummaryMap";
  MonitorElement * reportMe = dbe_->get(meName.str());
  
  MonitorElement * globalMe;
  
  //BARREL
  float barrelFactor = 0;
  
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
  
     barrelFactor = barrelFactor/5;
  

     float endcapFactor = 0;
     
     if(doEndcapCertification_){
       
       //Endcap
       for(int d = -numberDisk_ ; d<= numberDisk_; d++){
	 if (d==0) continue;
	 
	 meName.str("");
	 meName<<globalFolder_<<"/RPCChamberQuality_Ring_vs_Segment_Disk"<<d;
	 myMe = dbe_->get(meName.str());
	 
	 if(myMe){      
	   float diskFactor = 0;
	   
	   float sectorFactor[6]= {0,0,0,0,0,0};
	   
	   for (int i = 0 ;i <6;i++){
	     int firstSeg = (i *6 )+1;
	     int lastSeg = firstSeg +6;
	     int rollInSector = 0; 
	     for(int seg = firstSeg; seg< lastSeg ; seg++){
	       
	       for(int y = 1;y<=myMe->getNbinsY(); y++){
		 rollInSector++;
		 if(myMe->getBinContent(seg,y) == PARTIALLY_DEAD) sectorFactor[i]+=0.8;
		 else if(myMe->getBinContent(seg,y) == DEAD )sectorFactor[i]+=0;
		 else sectorFactor[i]+=1;	
		 
	       }
	     }
	     sectorFactor[i] = sectorFactor[i]/rollInSector;
	   }//end loop on Sectors
	   
	   
	   for (int sec = 0 ; sec<6; sec++){
	     diskFactor += sectorFactor[sec];	
	     if(reportMe)	{
	       if (d<0) reportMe->setBinContent(d+5, sec+1 , sectorFactor[sec]);
	       else  reportMe->setBinContent(d+11, sec+1 , sectorFactor[sec]);
	     } 	 
	   }
	   
	   diskFactor = diskFactor/6;
	   
	   meName.str("");
	   meName<<eventInfoPath_ + "/reportSummaryContents/RPC_Disk"<<d; 
	   globalMe=dbe_->get(meName.str());
	   if(globalMe) globalMe->Fill(diskFactor);
	   
	   endcapFactor += diskFactor;
	 }//end loop on disks
	 
       }
       
       endcapFactor=endcapFactor/ (numberDisk_ * 2);
       
     } 
     
     //Fill repor summary
     float rpcFactor = barrelFactor;
     if(doEndcapCertification_){ rpcFactor =  ( barrelFactor + endcapFactor)/2; }
     
     globalMe = dbe_->get(eventInfoPath_ +"/reportSummary"); 
     if(globalMe) globalMe->Fill(rpcFactor);
     
     
}

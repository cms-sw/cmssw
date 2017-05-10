#include <sstream>
#include <DQM/RPCMonitorClient/interface/RPCChamberQuality.h>
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"
#include "DQM/RPCMonitorDigi/interface/utils.h"
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//DataFormats
#include <DataFormats/MuonDetId/interface/RPCDetId.h>

const std::string RPCChamberQuality::xLabels_[7] = {"Good", "OFF", "Nois.St","Nois.Ch","Part.Dead","Dead","Bad.Shape"};
const std::string RPCChamberQuality::regions_[3] = {"EndcapNegative","Barrel","EndcapPositive"};

RPCChamberQuality::RPCChamberQuality(const edm::ParameterSet& ps ){
  edm::LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Constructor";
  
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 5);

  std::string subsystemFolder = ps.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  std::string recHitTypeFolder = ps.getUntrackedParameter<std::string>("RecHitTypeFolder", "AllHits");
  std::string summaryFolder = ps.getUntrackedParameter<std::string>("SummaryFolder", "SummaryHistograms");

  summaryDir_ =  subsystemFolder +"/"+  recHitTypeFolder +"/"+ summaryFolder ;
  prefixDir_ =  subsystemFolder +"/"+  recHitTypeFolder  ;

  enableDQMClients_ = ps.getUntrackedParameter<bool> ("EnableRPCDqmClient",true); 

  minEvents = ps.getUntrackedParameter<int>("MinimumRPCEvents", 10000);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  useRollInfo_ = ps.getUntrackedParameter<bool> ("UseRollInfo",false); 
  offlineDQM_ = ps.getUntrackedParameter<bool> ("OfflineDQM",true); 
}

RPCChamberQuality::~RPCChamberQuality(){
  edm::LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Destructor ";
  if(!  enableDQMClients_ ) return;
  dbe_=0;
}

void RPCChamberQuality::beginJob(){
  edm::LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Begin job ";
  if(!  enableDQMClients_ ) return;
  dbe_ = edm::Service<DQMStore>().operator->();
}

void RPCChamberQuality::beginRun(const edm::Run& r, const edm::EventSetup& c){
  edm::LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Begin run";
  if(!  enableDQMClients_ ) return;
  
  init_ = false;  
  lumiCounter_ = prescaleFactor_ ;

  MonitorElement* me;
  dbe_->setCurrentFolder(summaryDir_);
  
  std::stringstream histoName;
  
  rpcdqm::utils rpcUtils;

  for (int r = 0 ; r < 3; r++){

  histoName.str("");
  histoName<<"RPCChamberQuality_"<<regions_[r]; 
  me = dbe_->get(summaryDir_+"/"+ histoName.str());
  if (0!=me)    dbe_->removeElement(me->getName());
  me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  7, 0.5, 7.5);
  
  for (int x = 1; x <8 ; x++) me->setBinLabel(x, xLabels_[x-1]);
  }


  histoName.str("");
  histoName<<"RPC_System_Quality_Overview"; 
  me = dbe_->get(summaryDir_+"/"+ histoName.str());
  if (0!=me)       dbe_->removeElement(me->getName());
  me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  7, 0.5, 7.5, 3, 0.5, 3.5);
  me->setBinLabel(1, "E+", 2);
  me->setBinLabel(2, "B", 2);
  me->setBinLabel(3, "E-", 2);
    
  for (int x = 1; x <8 ; x++) me->setBinLabel(x, xLabels_[x-1]);
    
  for(int w=-2; w<3;w++){//Loop on wheels
    
    histoName.str("");
    histoName<<"RPCChamberQuality_Roll_vs_Sector_Wheel"<<w;    
    me = dbe_->get(summaryDir_+"/"+ histoName.str());
    if (0!=me) dbe_->removeElement(me->getName());
    me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);

    rpcUtils.labelXAxisSector( me);
    rpcUtils.labelYAxisRoll(me, 0, w, useRollInfo_ );

    histoName.str("");
    histoName<<"RPCChamberQuality_Distribution_Wheel"<<w;    
    me=0;
    me = dbe_->get(summaryDir_+"/"+ histoName.str());
    if (0!=me )   dbe_->removeElement(me->getName());
    me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  7, 0.5, 7.5);

    for (int x = 1; x <8; x++) me->setBinLabel(x, xLabels_[x-1]);        
  }//end loop on wheels

  for(int d= -numberOfDisks_; d<= numberOfDisks_ ; d++) { // Loop on disk
    if(d==0) continue; 
      histoName.str("");
      histoName<<"RPCChamberQuality_Ring_vs_Segment_Disk"<<d;       //  2D histo for RPC Qtest
      me = 0;
      me = dbe_->get(summaryDir_+"/"+ histoName.str());
      if (0!=me) {
	dbe_->removeElement(me->getName());
      }
      me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  36, 0.5, 36.5, 6, 0.5, 6.5);
      rpcUtils.labelXAxisSegment(me);
      rpcUtils.labelYAxisRing(me, 2, useRollInfo_ );

      histoName.str("");
      histoName<<"RPCChamberQuality_Distribution_Disk"<<d;    
      me=0;
      me = dbe_->get(summaryDir_+"/"+ histoName.str());
      if (0!=me )   dbe_->removeElement(me->getName());
      me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  7, 0.5, 7.5);
      
      for (int x = 1; x <8 ; x++) me->setBinLabel(x, xLabels_[x-1]); 
  } 
}

void RPCChamberQuality::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context){} 

void RPCChamberQuality::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {}

void RPCChamberQuality::endRun(const edm::Run& r, const edm::EventSetup& c) {
  edm::LogVerbatim ("rpceventsummary") <<"[RPCChamberQuality]: End Job, performing DQM client operation";
  if(!  enableDQMClients_ ) return;
  this->fillMonitorElements();

}

void RPCChamberQuality::fillMonitorElements() {

  std::stringstream meName;
   
  meName.str("");
  meName<<prefixDir_<<"/RPCEvents"; 
  int rpcEvents=minEvents;
  RpcEvents = dbe_->get(meName.str());
  
  if(RpcEvents) rpcEvents= (int)RpcEvents->getBinContent(1);
  
  if(rpcEvents >= minEvents){

    init_ = true;
    
    MonitorElement * summary[3];
    
    for(int r = 0 ; r < 3 ; r++) {    
      meName.str("");
      meName<<summaryDir_<<"/RPCChamberQuality_"<<RPCChamberQuality::regions_[r]; 
      summary[r] = dbe_ -> get(meName.str());
      
      if( summary[r] != 0 ) summary[r]->Reset();
    }
    
    //Barrel
    for (int wheel=-2; wheel<3; wheel++) { // loop by Wheels
      meName.str("");
      meName<<"Roll_vs_Sector_Wheel"<<wheel;
      
      this->performeClientOperation(meName.str(), 0 , summary[1]);
    } // loop by Wheels
    
    
    // Endcap
    for(int i=-3; i<4; i++) {//loop on Disks
      if(i==0) continue;
      
      meName.str("");
      meName<<"Ring_vs_Segment_Disk"<<i;

      if(i<0) this->performeClientOperation(meName.str(), -1 , summary[0]);
      else this->performeClientOperation(meName.str(), 1 , summary[2]);
    }//loop on Disks
      
    MonitorElement * RpcOverview = NULL;
    meName.str("");
    meName<<summaryDir_<<"/RPC_System_Quality_Overview"; 
    RpcOverview = dbe_ -> get(meName.str());
    RpcOverview->Reset();

    if(RpcOverview) {//Fill Overview ME
      for(int r = 0 ; r< 3; r++) {
	if (summary[r] == 0 ) continue;
	double entries = summary[r]->getEntries();
	if(entries == 0) continue;
	for (int x = 1; x <= 7; x++) {
	  RpcOverview->setBinContent(x,r+1,(summary[r]->getBinContent(x)/entries));
       } 
      } 
    } //loop by LimiBloks
  }
} 

void RPCChamberQuality::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {  

  if(!enableDQMClients_ ) return;

  if(offlineDQM_) return;

  if(!init_ ) {
    this->fillMonitorElements();
    return;
  }

  lumiCounter_++;
  
  if (lumiCounter_%prescaleFactor_ != 0) return;
  
  this->fillMonitorElements();
  

}


void RPCChamberQuality::performeClientOperation(std::string MESufix, int region, MonitorElement * quality){


  MonitorElement * RCQ=NULL;  
  MonitorElement * RCQD=NULL; 
  
  MonitorElement * DEAD=NULL;  
  MonitorElement * CLS=NULL;
  MonitorElement * MULT=NULL;
  MonitorElement * NoisySt=NULL;
  MonitorElement * Chip=NULL;
  MonitorElement * HV=NULL;
  MonitorElement * LV=NULL;
  std::stringstream meName; 

  meName.str("");
  meName<<summaryDir_<<"/RPCChamberQuality_"<<MESufix; 
  RCQ = dbe_ -> get(meName.str());
  //  if (RCQ)  RCQ->Reset();


  int pos = MESufix.find_last_of("_");
  meName.str("");
  meName<<summaryDir_<<"/RPCChamberQuality_Distribution"<<MESufix.substr(pos); 
  RCQD = dbe_ -> get(meName.str());
 if (RCQD) RCQD->Reset();
  
  //get HV Histo
  meName.str("");                        
  meName<<summaryDir_<<"/HVStatus_"<<MESufix;
  HV = dbe_ -> get(meName.str());	
  //get LV Histo
  meName.str("");                     
  meName<<summaryDir_<<"/LVStatus_"<<MESufix; 
  LV = dbe_ -> get(meName.str());
  //Dead 
  meName.str("");
  meName << summaryDir_<<"/DeadChannelFraction_"<<MESufix;
  DEAD = dbe_->get(meName.str());
  //ClusterSize
  meName.str("");
  meName<<summaryDir_<<"/ClusterSizeIn1Bin_"<<MESufix;
  CLS = dbe_ -> get(meName.str());
  //NoisyStrips
  meName.str("");
  meName<<summaryDir_<<"/RPCNoisyStrips_"<<MESufix;
  NoisySt = dbe_ -> get(meName.str());
  //Multiplicity
  meName.str("");
  meName<<summaryDir_<<"/NumberOfDigi_Mean_"<<MESufix;
  MULT = dbe_ -> get(meName.str());
  //Asymetry
  meName.str("");
  meName<<summaryDir_<<"/AsymmetryLeftRight_"<<MESufix;
  Chip = dbe_ -> get(meName.str());		   
  
  int xBinMax, yBinMax;    

  if (region != 0) xBinMax = 37;
  else xBinMax = 13;

  for(int x=1; x<xBinMax; x++) {
    if (region != 0 )  {
      yBinMax = 7;
    }else {  
      if(x==4) yBinMax=22;
      else if(x==9 || x==11) yBinMax=16;
      else yBinMax=18;
    }  
    for(int y=1; y<yBinMax; y++) {
      int hv=1;
      int lv=1;
      float dead =0;
      float firstbin= 0;
      float noisystrips = 0;
      float mult = 0;
      float asy = 0;
      chamberQualityState chamberState = GoodState;
       
      if(HV) hv = (int)HV ->getBinContent(x,y);
      if(LV) lv = (int)LV ->getBinContent(x,y);
        
      if( hv!=1 || lv!=1) { 
	chamberState = OffState;
      }else {
	if(DEAD) dead= DEAD -> getBinContent(x,y);
	if(dead>=0.80 ) {  
	  chamberState = DeadState;
	}else if (0.33<=dead && dead<0.80 ){
	  chamberState = PartiallyDeadState;
	}else {        
	  if(CLS ) firstbin = CLS -> getBinContent(x,y);
	  if(firstbin >= 0.88) {
	    chamberState = NoisyStripState;
	  } else {   
	    if(NoisySt)  noisystrips = NoisySt -> getBinContent(x,y);
	    if (noisystrips > 0){ 
	      chamberState = NoisyStripState;
	    }else {  
	      if(MULT) mult = MULT -> getBinContent(x,y);
	      if(mult>=6) {  
		chamberState = NoisyRollState;
	      }else {  
		if (Chip) asy = Chip->getBinContent(x,y);
		if(asy>0.35) {  
		  chamberState  = BadShapeState;
		}else {  
		  chamberState  = GoodState;
		}
	      }
	    } 
	  }
	}
      }
      if (RCQ)  RCQ -> setBinContent(x,y, chamberState);
      if (RCQD)   RCQD -> Fill(chamberState); 
      if (quality)   quality->Fill(chamberState); 
    }
  }
  return;
} 


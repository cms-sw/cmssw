/*  \author Anna Cimmino*/
//#include <cmath>
#include <sstream>
#include <DQM/RPCMonitorClient/interface/RPCOccupancyTest.h>
#include "DQM/RPCMonitorDigi/interface/utils.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

RPCOccupancyTest::RPCOccupancyTest(const edm::ParameterSet& ps ){
  edm::LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: Constructor";
  
  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
  testMode_ = ps.getUntrackedParameter<bool>("testMode", false);
  useRollInfo_ = ps.getUntrackedParameter<bool>("useRollInfo_", false);

  std::string subsystemFolder = ps.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  std::string recHitTypeFolder= ps.getUntrackedParameter<std::string>("RecHitTypeFolder", "AllHits");
   
  prefixDir_ =   subsystemFolder+ "/"+ recHitTypeFolder;
 
}

RPCOccupancyTest::~RPCOccupancyTest(){
  dbe_=0;
}

void RPCOccupancyTest::beginJob(DQMStore * dbe, std::string workingFolder){
 edm::LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: Begin job ";
 globalFolder_ =  workingFolder;
 dbe_=dbe;
}

void RPCOccupancyTest::endRun(const edm::Run& r, const edm::EventSetup& c){
 edm::LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: End run";
 }
 
void RPCOccupancyTest::getMonitorElements(std::vector<MonitorElement *> & meVector, std::vector<RPCDetId> & detIdVector){
  
 //Get Occupancy  ME for each roll
  for (unsigned int i = 0 ; i<meVector.size(); i++){
    
    bool flag= false;
    
    DQMNet::TagList tagList;
    tagList = meVector[i]->getTags();
    DQMNet::TagList::iterator tagItr = tagList.begin();
    
    while (tagItr != tagList.end() && !flag ) {
      if((*tagItr) ==  rpcdqm::OCCUPANCY)
	flag= true;      
      tagItr++;
    }
    
    if(flag){
      myOccupancyMe_.push_back(meVector[i]);
      myDetIds_.push_back(detIdVector[i]);
    }
  }
}

void RPCOccupancyTest::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context){} 

void RPCOccupancyTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {}

void RPCOccupancyTest::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {}

void RPCOccupancyTest::clientOperation(edm::EventSetup const& iSetup) {

  edm::LogVerbatim ("rpceventsummary") <<"[RPCOccupancyTest]: Client Operation";

   MonitorElement * RPCEvents = dbe_->get(prefixDir_ +"/RPCEvents");  
   if(RPCEvents == 0 ){rpcevents_ = 0;}
   else{
     rpcevents_ = RPCEvents ->getBinContent(1);
   }
     
 //Loop on MEs
  for (unsigned int  i = 0 ; i<myOccupancyMe_.size();i++){
    this->fillGlobalME(myDetIds_[i],myOccupancyMe_[i]);
  }//End loop on MEs
}

void RPCOccupancyTest::endJob(void) {}
void RPCOccupancyTest::beginRun(const edm:: Run& r, const edm::EventSetup& c) {

 MonitorElement* me;
 dbe_->setCurrentFolder( globalFolder_);

 std::stringstream histoName;
 rpcdqm::utils rpcUtils;

 histoName.str("");
 histoName<<"Barrel_OccupancyByStations_Normalized";
 me = dbe_->get( globalFolder_+"/"+ histoName.str());
 if ( 0!=me  ) {
   dbe_->removeElement(me->getName());
 }
 Barrel_OccBySt = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  4, 0.5, 4.5);
 Barrel_OccBySt -> setBinLabel(1, "St1", 1);
 Barrel_OccBySt -> setBinLabel(2, "St2", 1);
 Barrel_OccBySt -> setBinLabel(3, "St3", 1);
 Barrel_OccBySt -> setBinLabel(4, "St4", 1);
 
 
 histoName.str("");
 histoName<<"EndCap_OccupancyByRings_Normalized";
 me = dbe_->get( globalFolder_+"/"+ histoName.str());
 if ( 0!=me  ) {
   dbe_->removeElement(me->getName());
 }
 EndCap_OccByRng = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  4, 0.5, 4.5);
 EndCap_OccByRng -> setBinLabel(1, "E+/R3", 1);
 EndCap_OccByRng -> setBinLabel(2, "E+/R2", 1);
 EndCap_OccByRng -> setBinLabel(3, "E-/R2", 1);
 EndCap_OccByRng -> setBinLabel(4, "E-/R3", 1);

//  histoName.str("");
//  histoName<<"EndCap_OccupancyByDisksAndRings_Normalized";
//  me = dbe_->get( globalFolder_+"/"+ histoName.str());
//  if ( 0!=me  ) {
//    dbe_->removeElement(me->getName());
//  }
//  EndCap_OccByDisk = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  12, 0, 12);
//  EndCap_OccByDisk -> setBinLabel(1, "YE-3/R2", 1);
//  EndCap_OccByDisk -> setBinLabel(2, "YE-2/R2", 1);
//  EndCap_OccByDisk -> setBinLabel(3, "YE-1/R2", 1);
//  EndCap_OccByDisk -> setBinLabel(4, "YE+1/R2", 1);
//  EndCap_OccByDisk -> setBinLabel(5, "YE+2/R2", 1);
//  EndCap_OccByDisk -> setBinLabel(6, "YE+3/R2", 1);

//  EndCap_OccByDisk -> setBinLabel(7, "YE-3/R3", 1);
//  EndCap_OccByDisk -> setBinLabel(8, "YE-2/R3", 1);
//  EndCap_OccByDisk -> setBinLabel(9, "YE-1/R3", 1);
//  EndCap_OccByDisk -> setBinLabel(10, "YE+1/R3", 1);
//  EndCap_OccByDisk -> setBinLabel(11, "YE+2/R3", 1);
//  EndCap_OccByDisk -> setBinLabel(12, "YE+3/R3", 1);
 
  for (int w = -2; w<=2; w++ ){//loop on wheels
 
    histoName.str("");
    histoName<<"AsymmetryLeftRight_Roll_vs_Sector_Wheel"<<w;
    me = 0;
    me = dbe_->get( globalFolder_+"/"+ histoName.str());
    if ( 0!=me  ) {
      dbe_->removeElement(me->getName());
    }
    
    AsyMeWheel[w+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
    
    rpcUtils.labelXAxisSector(AsyMeWheel[w+2]);
    rpcUtils.labelYAxisRoll(AsyMeWheel[w+2], 0, w,  useRollInfo_);
  
    
    if(testMode_){
  
      histoName.str("");
      histoName<<"OccupancyNormByEvents_Wheel"<<w;
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      
      NormOccupWheel[w+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
      
      rpcUtils.labelXAxisSector(  NormOccupWheel[w+2]);
      rpcUtils.labelYAxisRoll(  NormOccupWheel[w+2], 0, w,  useRollInfo_);
      
      
      histoName.str("");
      histoName<<"AsymmetryLeftRight_Distribution_Wheel"<<w;  
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      AsyMeDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, -0.1, 1.1);
      
      histoName.str("");
      histoName<<"OccupancyNormByEvents_Distribution_Wheel"<<w;   
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      NormOccupDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.0, 0.205);
    }
  }//end Barrel
  
  for(int d = -numberOfDisks_; d<=numberOfDisks_; d++ ){

    if (d == 0)continue;
    
    int offset = numberOfDisks_;
    if (d>0) offset --; //used to skip case equale to zero
    
    histoName.str("");
    histoName<<"AsymmetryLeftRight_Ring_vs_Segment_Disk"<<d;
    me = 0;
    me = dbe_->get( globalFolder_+"/"+ histoName.str());
    if ( 0!=me  ) {
      dbe_->removeElement(me->getName());
    }
    
    AsyMeDisk[d+offset] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5);
    
    rpcUtils.labelXAxisSegment(AsyMeDisk[d+offset]);
    rpcUtils.labelYAxisRing(AsyMeDisk[d+offset], numberOfRings_,  useRollInfo_);
    
   
    
    if(testMode_){
   
      histoName.str("");
      histoName<<"OccupancyNormByEvents_Disk"<<d;
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      
      NormOccupDisk[d+offset] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5);
      
      rpcUtils.labelXAxisSegment(NormOccupDisk[d+offset]);
      rpcUtils.labelYAxisRing( NormOccupDisk[d+offset],numberOfRings_,  useRollInfo_);
      
      histoName.str("");
      histoName<<"AsymmetryLeftRight_Distribution_Disk"<<d;      
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      AsyMeDDisk[d+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, -0.1, 1.1);
      
      
      
      histoName.str("");
      histoName<<"OccupancyNormByEvents_Distribution_Disk"<<d;  
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      NormOccupDDisk[d+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.0, 0.205);
    }
  }//End loop on Endcap
}


void RPCOccupancyTest::fillGlobalME(RPCDetId & detId, MonitorElement * myMe){
  

if (!myMe) return;
    
    MonitorElement * AsyMe=NULL;      //Left Right Asymetry 
    MonitorElement * AsyMeD=NULL; 
    MonitorElement * NormOccup=NULL;
    MonitorElement * NormOccupD=NULL;
       
    if(detId.region() ==0){
      AsyMe= AsyMeWheel[detId.ring()+2];
      if(testMode_){
	NormOccup=NormOccupWheel[detId.ring()+2];
	AsyMeD= AsyMeDWheel[detId.ring()+2];
	NormOccupD=NormOccupDWheel[detId.ring()+2];
      }

    }else{

      if( -detId.station() +  numberOfDisks_ >= 0 ){
	
	if(detId.region()<0){
	  AsyMe= AsyMeDisk[-detId.station()  + numberOfDisks_];
	  if(testMode_){
	    NormOccup=NormOccupDisk[-detId.station() + numberOfDisks_];
	    AsyMeD= AsyMeDDisk[-detId.station() + numberOfDisks_];	  
	    NormOccupD=NormOccupDDisk[-detId.station() + numberOfDisks_];
	  }
	}else{
	  AsyMe= AsyMeDisk[detId.station() + numberOfDisks_-1];
	  if(testMode_){
	    NormOccup=NormOccupDisk[detId.station() + numberOfDisks_-1];
	    AsyMeD= AsyMeDDisk[detId.station() + numberOfDisks_-1];
	    NormOccupD=NormOccupDDisk[detId.station() + numberOfDisks_-1];
	  }
	}
      }
    }

 
    int xBin,yBin;
    if(detId.region()==0){//Barrel
      xBin= detId.sector();
      rpcdqm::utils rollNumber;
      yBin = rollNumber.detId2RollNr(detId);
    }else{//Endcap
      //get segment number
      RPCGeomServ RPCServ(detId);
      xBin = RPCServ.segment();
      (numberOfRings_ == 3 ? yBin= detId.ring()*3-detId.roll()+1 : yBin= (detId.ring()-1)*3-detId.roll()+1);
    }
    
	
    int stripInRoll=myMe->getNbinsX();
    float FOccupancy=0;
    float BOccupancy=0;
    
    float  totEnt =  myMe->getEntries();
    for(int strip = 1 ; strip<=stripInRoll; strip++){
      if(strip<=stripInRoll/2) FOccupancy+=myMe->getBinContent(strip);
      else  BOccupancy+=myMe->getBinContent(strip);
    }
	    

    float asym = 0;
    if(totEnt != 0 ) asym =  fabs((FOccupancy - BOccupancy )/totEnt);
    
    if(AsyMe)  AsyMe->setBinContent(xBin,yBin,asym);


	
    float normoccup = 1;
    if(rpcevents_ != 0) normoccup = (totEnt/rpcevents_);
   
    if(testMode_){
      if(NormOccup)  NormOccup->setBinContent(xBin,yBin, normoccup);
      if(AsyMeD) AsyMeD->Fill(asym);
      if(NormOccupD) NormOccupD->Fill(normoccup);
    }    
   

    if(detId.region()==0) {
      if(Barrel_OccBySt)Barrel_OccBySt -> Fill(detId.station(), normoccup);
    }else if(detId.region()==1) {
      if(detId.ring()==3) {
	EndCap_OccByRng -> Fill(1, normoccup);
      } else {
	EndCap_OccByRng -> Fill(2, normoccup);
      }
    } else {
      if(detId.ring()==3) {
	EndCap_OccByRng -> Fill(4, normoccup);
      }else {
	EndCap_OccByRng -> Fill(3, normoccup);
      }
    }

}






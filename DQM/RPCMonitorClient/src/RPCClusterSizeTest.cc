#include <DQM/RPCMonitorClient/interface/RPCClusterSizeTest.h>
#include "DQM/RPCMonitorDigi/interface/utils.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// //Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

RPCClusterSizeTest::RPCClusterSizeTest(const edm::ParameterSet& ps ){
  edm::LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: Constructor";
  
  prescaleFactor_ =  ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
  testMode_ = ps.getUntrackedParameter<bool>("testMode", false);
  useRollInfo_ = ps.getUntrackedParameter<bool>("useRollInfo", false);

  resetMEArrays();
}

RPCClusterSizeTest::~RPCClusterSizeTest(){ dbe_=0;}

void RPCClusterSizeTest::beginJob(DQMStore *  dbe, std::string workingFolder){
  edm::LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: Begin job ";

  globalFolder_  = workingFolder;
  dbe_ = dbe;
}

void RPCClusterSizeTest::endRun(const edm::Run& r, const edm::EventSetup& c){
  edm::LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: End run";
}

void RPCClusterSizeTest::getMonitorElements(std::vector<MonitorElement *> & meVector, std::vector<RPCDetId> & detIdVector){
    
 
 //Get  ME for each roll
 for (unsigned int i = 0 ; i<meVector.size(); i++){

   bool flag= false;
   
   DQMNet::TagList tagList;
   tagList = meVector[i]->getTags();
   DQMNet::TagList::iterator tagItr = tagList.begin();

   while (tagItr != tagList.end() && !flag ) {
     if((*tagItr) ==  rpcdqm::CLUSTERSIZE)
       flag= true;
   
     tagItr++;
   }
   
   if(flag){
     myClusterMe_.push_back(meVector[i]);
     myDetIds_.push_back(detIdVector[i]);
   }
 }

}

void RPCClusterSizeTest::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context){} 

void RPCClusterSizeTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {}

void RPCClusterSizeTest::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {} 

void RPCClusterSizeTest::clientOperation(edm::EventSetup const& iSetup) {
  
  edm::LogVerbatim ("rpceventsummary") <<"[RPCClusterSizeTest]:Client Operation";
  
  //check some statements and prescale Factor
  if(myClusterMe_.size()==0 || myDetIds_.size()==0)return;
        
  MonitorElement * CLS   = NULL;  // ClusterSize in 1 bin, Roll vs Sector
  MonitorElement * CLSD  = NULL;  // ClusterSize in 1 bin, Distribution
  MonitorElement * MEAN  = NULL;  // Mean ClusterSize, Roll vs Sector
  MonitorElement * MEAND = NULL;  // Mean ClusterSize, Distribution
  
  
  std::stringstream meName;
  RPCDetId detId;
  MonitorElement * myMe;

  
  //Loop on chambers
  for (unsigned int  i = 0 ; i<myClusterMe_.size();i++){
    
    myMe = myClusterMe_[i];
    if (!myMe || myMe->getEntries()==0 )continue;

    
    detId=myDetIds_[i];
    
    
    if (detId.region()==0){

      CLS = CLSWheel[detId.ring()+2];
      MEAN = MEANWheel[detId.ring()+2];
      if(testMode_){
	CLSD = CLSDWheel[detId.ring()+2];
      	MEAND = MEANDWheel[detId.ring()+2];
      }
    }else {
      
      if(((detId.station() * detId.region() ) + numberOfDisks_) >= 0 ){
	
	if(detId.region()<0){
	  CLS=CLSDisk[(detId.station() * detId.region() ) + numberOfDisks_];
	  MEAN=  MEANDisk[(detId.station() * detId.region() ) + numberOfDisks_];
	  if(testMode_){
	    CLSD = CLSDDisk[(detId.station() * detId.region() ) + numberOfDisks_];
	    MEAND= MEANDDisk[(detId.station() * detId.region() ) + numberOfDisks_];
	  }
	}else{
	  CLS=CLSDisk[(detId.station() * detId.region() ) + numberOfDisks_ -1];
	  MEAN= MEANDisk[(detId.station() * detId.region() ) + numberOfDisks_-1];
	  if(testMode_){
	    CLSD = CLSDDisk[(detId.station() * detId.region() ) + numberOfDisks_-1];
	    MEAND= MEANDDisk[(detId.station() * detId.region() ) + numberOfDisks_-1];
	  }
	}
      }
      
    }

  
    int xBin,yBin;
    
    if (detId.region()==0){//Barrel
      
      rpcdqm::utils rollNumber;
      yBin = rollNumber.detId2RollNr(detId);
      xBin = detId.sector();
    }else {//Endcap
      
      //get segment number
      RPCGeomServ RPCServ(detId);
      xBin = RPCServ.segment();
      (numberOfRings_ == 3 ? yBin= detId.ring()*3-detId.roll()+1 : yBin= (detId.ring()-1)*3-detId.roll()+1);
    }
    
    // Normalization -> # of Entries in first Bin normalaized by total Entries
    
    float NormCLS = myMe->getBinContent(1)/myMe->getEntries();
    float meanCLS = myMe->getMean();
    
    if (CLS)  CLS -> setBinContent(xBin,yBin, NormCLS);
    if(MEAN)   MEAN -> setBinContent(xBin, yBin, meanCLS);
 
    if(testMode_){
      if(MEAND) MEAND->Fill(meanCLS);
      if(CLSD)   CLSD->Fill(NormCLS);
    }
    
  }//End loop on chambers
} 

void  RPCClusterSizeTest::endJob(void) {}

void RPCClusterSizeTest::resetMEArrays(void) {
  memset((void*) CLSWheel, 0, sizeof(MonitorElement*)*kWheels);
  memset((void*) CLSDWheel, 0, sizeof(MonitorElement*)*kWheels);
  memset((void*) MEANWheel, 0, sizeof(MonitorElement*)*kWheels);
  memset((void*) MEANDWheel, 0, sizeof(MonitorElement*)*kWheels);

  memset((void*) CLSDisk, 0, sizeof(MonitorElement*)*kDisks);
  memset((void*) CLSDDisk, 0, sizeof(MonitorElement*)*kDisks);
  memset((void*) MEANDisk, 0, sizeof(MonitorElement*)*kDisks);
  memset((void*) MEANDDisk, 0, sizeof(MonitorElement*)*kDisks);
}


void  RPCClusterSizeTest::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  resetMEArrays();
  
  MonitorElement* me;
  dbe_->setCurrentFolder(globalFolder_);

  std::stringstream histoName;

  rpcdqm::utils rpcUtils;

  // Loop over wheels
  for (int w = -2; w <= 2; w++) {
    histoName.str("");   
    histoName<<"ClusterSizeIn1Bin_Roll_vs_Sector_Wheel"<<w;       // ClusterSize in first bin norm. by Entries (2D Roll vs Sector)       
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me ) {
      dbe_->removeElement(me->getName());
    }
    
    CLSWheel[w+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
    rpcUtils.labelXAxisSector(  CLSWheel[w+2]);
    rpcUtils.labelYAxisRoll(   CLSWheel[w+2], 0, w ,useRollInfo_);
    
    
    histoName.str("");
    histoName<<"ClusterSizeMean_Roll_vs_Sector_Wheel"<<w;       // Avarage ClusterSize (2D Roll vs Sector)   
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me) {
      dbe_->removeElement(me->getName());
    }
    
    MEANWheel[w+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
    
    rpcUtils.labelXAxisSector(  MEANWheel[w+2]);
    rpcUtils.labelYAxisRoll(MEANWheel[w+2], 0, w,useRollInfo_ );
    
    if(testMode_){
      histoName.str("");
      histoName<<"ClusterSizeIn1Bin_Distribution_Wheel"<<w;       //  ClusterSize in first bin, distribution
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me ) {
	dbe_->removeElement(me->getName());
      }
      CLSDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, 0.0, 1.0);
      
      
      histoName.str("");
      histoName<<"ClusterSizeMean_Distribution_Wheel"<<w;       //  Avarage ClusterSize Distribution
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me){
	dbe_->removeElement(me->getName());
      }
      MEANDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.5, 10.5);
    }
  }//end loop on wheels


  for (int d = -numberOfDisks_;  d <= numberOfDisks_; d++) {
    if (d == 0)
      continue;
  //Endcap
    int offset = numberOfDisks_;
    if (d>0) offset--;

    histoName.str("");   
    histoName<<"ClusterSizeIn1Bin_Ring_vs_Segment_Disk"<<d;       // ClusterSize in first bin norm. by Entries (2D Roll vs Sector)   
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me){
      dbe_->removeElement(me->getName());
    }
    
    CLSDisk[d+offset] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5); 
    rpcUtils.labelXAxisSegment(CLSDisk[d+offset]);
    rpcUtils.labelYAxisRing(CLSDisk[d+offset], numberOfRings_,useRollInfo_ );
   
    if(testMode_){
      histoName.str("");
      histoName<<"ClusterSizeIn1Bin_Distribution_Disk"<<d;       //  ClusterSize in first bin, distribution
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me){
	dbe_->removeElement(me->getName());
      }
      CLSDDisk[d+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, 0.0, 1.0);
      
      histoName.str("");
      histoName<<"ClusterSizeMean_Distribution_Disk"<<d;       //  Avarage ClusterSize Distribution
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me){
	dbe_->removeElement(me->getName());
      }
      MEANDDisk[d+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.5, 10.5);
      
    }
    
    histoName.str("");
    histoName<<"ClusterSizeMean_Ring_vs_Segment_Disk"<<d;       // Avarage ClusterSize (2D Roll vs Sector)   
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me){
      dbe_->removeElement(me->getName());
    }
    
    MEANDisk[d+offset] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5);
    rpcUtils.labelXAxisSegment(MEANDisk[d+offset]);
    rpcUtils.labelYAxisRing(MEANDisk[d+offset], numberOfRings_ ,useRollInfo_);
 }
}

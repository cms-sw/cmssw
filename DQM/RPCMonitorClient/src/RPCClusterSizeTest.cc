#include <DQM/RPCMonitorClient/interface/RPCClusterSizeTest.h>
#include "DQM/RPCMonitorDigi/interface/utils.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// //Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"


using namespace edm;
using namespace std;
RPCClusterSizeTest::RPCClusterSizeTest(const ParameterSet& ps ){
  LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: Constructor";
  
  prescaleFactor_ =  ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  globalFolder_ = ps.getUntrackedParameter<string>("RPCGlobalFolder", "RPC/RecHits/SummaryHistograms/");
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 3);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
}

RPCClusterSizeTest::~RPCClusterSizeTest(){ dbe_=0;}

void RPCClusterSizeTest::beginJob(DQMStore *  dbe){
  LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: Begin job ";
  dbe_ = dbe;
}

void RPCClusterSizeTest::beginRun(const Run& r, const EventSetup& c,vector<MonitorElement *> meVector, vector<RPCDetId> detIdVector){
  LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: Begin run";
  

  MonitorElement* me;
  dbe_->setCurrentFolder(globalFolder_);

  stringstream histoName;

  rpcdqm::utils rpcUtils;

  int limit = numberOfDisks_;
  if(numberOfDisks_ < 2) limit = 2;
  
  for (int w = -1 * limit; w<=limit;w++ ){//loop on wheels and disks
    if (w>-3 && w<3){//wheels
      histoName.str("");   
      histoName<<"ClusterSizeIn1Bin_Roll_vs_Sector_Wheel"<<w;       // ClusterSize in first bin norm. by Entries (2D Roll vs Sector)       
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me ) {
	dbe_->removeElement(me->getName());
      }
      
      CLSWheel[w+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
      rpcUtils.labelXAxisSector(  CLSWheel[w+2]);
      rpcUtils.labelYAxisRoll(   CLSWheel[w+2], 0, w);
      
      histoName.str("");
      histoName<<"ClusterSizeIn1Bin_Distribution_Wheel"<<w;       //  ClusterSize in first bin, distribution
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me ) {
	dbe_->removeElement(me->getName());
      }
      CLSDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, 0.0, 1.0);
      

      histoName.str("");
      histoName<<"ClusterSizeMean_Roll_vs_Sector_Wheel"<<w;       // Avarage ClusterSize (2D Roll vs Sector)   
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me) {
	dbe_->removeElement(me->getName());
      }
      
      MEANWheel[w+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
 
      rpcUtils.labelXAxisSector(  MEANWheel[w+2]);
      rpcUtils.labelYAxisRoll(MEANWheel[w+2], 0, w);

      histoName.str("");
      histoName<<"ClusterSizeMean_Distribution_Wheel"<<w;       //  Avarage ClusterSize Distribution
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me){
	dbe_->removeElement(me->getName());
      }
      MEANDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.5, 10.5);
    }//end loop on wheels

    if (w == 0 || w< (-1 * numberOfDisks_) || w > numberOfDisks_)continue;
    //Endcap
    int offset = numberOfDisks_;
    if (w>0) offset --; //used to skip case equale to zero

    histoName.str("");   
    histoName<<"ClusterSizeIn1Bin_Roll_vs_Sector_Disk"<<w;       // ClusterSize in first bin norm. by Entries (2D Roll vs Sector)   
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me){
      dbe_->removeElement(me->getName());
    }
    
    CLSDisk[w+offset] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5 );
    rpcUtils.labelXAxisSegment(CLSDisk[w+offset]);
    rpcUtils.labelYAxisRing(CLSDisk[w+offset], numberOfRings_);
    
    histoName.str("");
    histoName<<"ClusterSizeIn1Bin_Distribution_Disk"<<w;       //  ClusterSize in first bin, distribution
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me){
      dbe_->removeElement(me->getName());
    }
    CLSDDisk[w+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, 0.0, 1.0);
    
    
    histoName.str("");
    histoName<<"ClusterSizeMean_Roll_vs_Sector_Disk"<<w;       // Avarage ClusterSize (2D Roll vs Sector)   
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me){
      dbe_->removeElement(me->getName());
    }
    
    MEANDisk[w+offset] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5);    
    rpcUtils.labelXAxisSegment(MEANDisk[w+offset]);
    rpcUtils.labelYAxisRing(MEANDisk[w+offset], numberOfRings_);
    
    histoName.str("");
    histoName<<"ClusterSizeMean_Distribution_Disk"<<w;       //  Avarage ClusterSize Distribution
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me){
      dbe_->removeElement(me->getName());
    }
    MEANDDisk[w+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.5, 10.5);
 }


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

void RPCClusterSizeTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCClusterSizeTest::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCClusterSizeTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCClusterSizeTest]: End of LS transition, performing DQM client operation";
  
  //check some statements and prescale Factor
  if(lumiSeg.id().luminosityBlock()%prescaleFactor_ != 0 || myClusterMe_.size()==0 || myDetIds_.size()==0)return;
        
  MonitorElement * CLS =NULL;          // ClusterSize in 1 bin, Roll vs Sector
  MonitorElement * CLSD =NULL;         // ClusterSize in 1 bin, Distribution
  MonitorElement * MEAN =NULL;         // Mean ClusterSize, Roll vs Sector
  MonitorElement * MEAND =NULL;        // Mean ClusterSize, Distribution
  
  
  stringstream meName;
  RPCDetId detId;
  MonitorElement * myMe;

  //clear

 //Clear Distributions
  int limit = numberOfDisks_ * 2;
  if(numberOfDisks_<2) limit = 5;
  for(int i =0 ; i<limit; i++){
    if(i < numberOfDisks_ * 2){
      MEANDDisk[i]->Reset();
      CLSDDisk[i]->Reset();
    }
    if(i<5){
      MEANDWheel[i]->Reset();
      CLSDWheel[i]->Reset();
    }
  }
  
  //Loop on chambers
  for (unsigned int  i = 0 ; i<myClusterMe_.size();i++){

    myMe = myClusterMe_[i];
    if (!myMe)continue;

    
    detId=myDetIds_[i];
    
    
    if (detId.region()==0){

      CLS=CLSWheel[detId.ring()+2];
      CLSD=CLSDWheel[detId.ring()+2];
      MEAN= MEANWheel[detId.ring()+2];
      MEAND=MEANDWheel[detId.ring()+2];
    }else {
      
      if((-detId.station() + numberOfDisks_) >= 0 ){
	
	if(detId.region()<0){
	  CLS=CLSDisk[-detId.station() + numberOfDisks_];
	  CLSD = CLSDDisk[-detId.station() + numberOfDisks_];
	  MEAN=  MEANDisk[-detId.station() + numberOfDisks_];
	  MEAND= MEANDDisk[-detId.station() + numberOfDisks_];
	}else{
	  CLS=CLSDisk[detId.station() + numberOfDisks_ -1];
	  CLSD = CLSDDisk[detId.station()+ numberOfDisks_-1];
	  MEAN= MEANDisk[detId.station() + numberOfDisks_-1];
	  MEAND= MEANDDisk[detId.station()+ numberOfDisks_-1];
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
  
    if(MEAND) MEAND->Fill(meanCLS);
    if(CLSD)   CLSD->Fill(NormCLS);
  }//End loop on chambers
}
 

void  RPCClusterSizeTest::endJob(void) {}
void  RPCClusterSizeTest::endRun(const Run& r, const EventSetup& c) {}

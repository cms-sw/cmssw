/*
 *  \author Anna Cimmino
 */
#include "DQM/RPCMonitorDigi/interface/utils.h"
#include <DQM/RPCMonitorClient/interface/RPCMultiplicityTest.h>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//DQMServices
# include "DQMServices/Core/interface/DQMNet.h"
// Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

#include <sstream>

RPCMultiplicityTest::RPCMultiplicityTest(const edm::ParameterSet& ps ){
  edm::LogVerbatim ("multiplicity") << "[RPCMultiplicityTest]: Constructor";
  useRollInfo_ = ps.getUntrackedParameter<bool>("UseRollInfo", false);
  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
  testMode_ = ps.getUntrackedParameter<bool>("testMode", false);
}

RPCMultiplicityTest::~RPCMultiplicityTest(){
  dbe_ = 0;
}


void RPCMultiplicityTest::beginJob(DQMStore *  dbe, std::string workingFolder ){
 edm::LogVerbatim ("multiplicity") << "[RPCMultiplicityTest]: Begin job";

 globalFolder_ =  workingFolder;
 dbe_=dbe;
}


void RPCMultiplicityTest::endRun(const edm::Run& r, const edm::EventSetup& iSetup){

  edm::LogVerbatim ("multiplicity") << "[RPCMultiplicityTest]: End run";
  
}
 
void RPCMultiplicityTest::getMonitorElements(std::vector<MonitorElement *> & meVector, std::vector<RPCDetId> &  detIdVector){

  //Get NumberOfDigi ME for each roll
  for (unsigned int i = 0 ; i<meVector.size(); i++){
    
    bool flag= false;
    
    DQMNet::TagList tagList;
    tagList = meVector[i]->getTags();
    DQMNet::TagList::iterator tagItr = tagList.begin();
    
    while (tagItr != tagList.end() && !flag ) {
      if((*tagItr) ==  rpcdqm::MULTIPLICITY)
	flag= true;
      
      tagItr++;
    }
    
    if(flag){
      myNumDigiMe_.push_back(meVector[i]);
      myDetIds_.push_back(detIdVector[i]);
    }
  }
}


void RPCMultiplicityTest::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) {}

void RPCMultiplicityTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c){}

void RPCMultiplicityTest::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {}

void RPCMultiplicityTest::clientOperation(edm::EventSetup const& iSetup) {

  edm::LogVerbatim ("multiplicity") <<"[RPCMultiplicityTest]: Client Operation";
 
  //Loop on MEs
  for (unsigned int  i = 0 ; i<myNumDigiMe_.size();i++){
    this->fillGlobalME(myDetIds_[i],myNumDigiMe_[i]);
  }//End loop on MEs
}
 
void RPCMultiplicityTest::beginRun(const edm::Run& r, const edm::EventSetup& c){

  MonitorElement* me=NULL;
  dbe_->setCurrentFolder(globalFolder_);
  
  std::stringstream histoName;
  
  rpcdqm::utils rpcUtils;
  
  for (int i = -2; i<=2;i++ ){//loop on wheels and disks
 
    histoName.str("");
    histoName<<"NumberOfDigi_Mean_Roll_vs_Sector_Wheel"<<i;
    me = 0;
    me = dbe_->get(globalFolder_ +"/"+ histoName.str());
    if ( 0!=me) {
      dbe_->removeElement(me->getName());
    }
    
    MULTWheel[i+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 12, 0.5, 12.5, 21, 0.5, 21.5);
    
    rpcUtils.labelXAxisSector( MULTWheel[i+2]);
    rpcUtils.labelYAxisRoll( MULTWheel[i+2], 0, i,useRollInfo_ );
    
    if(testMode_){
      histoName.str("");
      histoName<<"NumberOfDigi_Mean_Distribution_Wheel"<<i;
      me = 0;
      me = dbe_->get(globalFolder_ +"/"+ histoName.str());
      if ( 0!=me) {
	dbe_->removeElement(me->getName());
      }
      
      MULTDWheel[i+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 50.5);
    }
    
  }//end wheels

  for(int d = -numberOfDisks_; d<=numberOfDisks_; d++ ){  
    if (d == 0 )continue;
    
    int offset = numberOfDisks_;
    if (d>0) offset --; //used to skip case equale to zero
    
    histoName.str("");
    histoName<<"NumberOfDigi_Mean_Ring_vs_Segment_Disk"<<d;
    me = 0;
    me = dbe_->get(globalFolder_ +"/"+ histoName.str());
    if ( 0!=me) {
      dbe_->removeElement(me->getName());
    }
    MULTDisk[d+offset]   = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5);
    rpcUtils.labelXAxisSegment(MULTDisk[d+offset]);
    rpcUtils.labelYAxisRing(MULTDisk[d+offset], numberOfRings_,useRollInfo_ );
    
    if(testMode_){
      histoName.str("");
      histoName<<"NumberOfDigi_Mean_Distribution_Disk"<<d;
      me = 0;
      me = dbe_->get(globalFolder_ +"/"+ histoName.str());
      if ( 0!=me) {
	dbe_->removeElement(me->getName());
      }
      
      MULTDDisk[d+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 50.5);
    }
  }//end loop on wheels and disks


}

 void RPCMultiplicityTest::endJob(){}

void  RPCMultiplicityTest::fillGlobalME(RPCDetId & detId, MonitorElement * myMe){

  MonitorElement * MULT =NULL;
  MonitorElement * MULTD = NULL;

  if (detId.region()==0) {
    MULT = MULTWheel[detId.ring()+2];
    if(testMode_)   MULTD = MULTDWheel[detId.ring()+2];
  }else{
    if(-detId.station() + numberOfDisks_ >= 0 ){
    
      if(detId.region()<0){
      MULT = MULTDisk[-detId.station() + numberOfDisks_];
      if(testMode_)    MULTD = MULTDDisk[-detId.station()+ numberOfDisks_];
      }else{
	MULT = MULTDisk[detId.station()+ numberOfDisks_ -1];
   if(testMode_) 	MULTD = MULTDDisk[detId.station()+ numberOfDisks_-1];
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
    
    float mean = myMe->getMean();
    
    if(MULT)  MULT->setBinContent(xBin,yBin, mean );
    if(testMode_ && MULTD) MULTD->Fill(mean);

  
  
}

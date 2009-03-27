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

using namespace edm;
using namespace std;

RPCMultiplicityTest::RPCMultiplicityTest(const ParameterSet& ps ){
  LogVerbatim ("multiplicity") << "[RPCMultiplicityTest]: Constructor";

  globalFolder_ = ps.getUntrackedParameter<string>("RPCGlobalFolder", "RPC/RecHits/SummaryHistograms");
  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 3);

}

RPCMultiplicityTest::~RPCMultiplicityTest(){
  dbe_ = 0;
}


void RPCMultiplicityTest::beginJob(DQMStore *  dbe ){
 LogVerbatim ("multiplicity") << "[RPCMultiplicityTest]: Begin job";
 dbe_=dbe;
}


void RPCMultiplicityTest::beginRun(const Run& r, const EventSetup& iSetup,vector<MonitorElement *> meVector, vector<RPCDetId> detIdVector){

  edm::LogVerbatim ("multiplicity") << "[RPCMultiplicityTest]: Begin run";
  
  MonitorElement* me=NULL;
  dbe_->setCurrentFolder(globalFolder_);
  
  stringstream histoName;
  
  rpcdqm::utils rpcUtils;
  
  int limit = numberOfDisks_;
  if(numberOfDisks_ < 2) limit = 2;
  
  for (int i = -1 * limit; i<=limit;i++ ){//loop on wheels and disks
  
    if (i>-3 && i<3){//wheels  
      histoName.str("");
      histoName<<"NumberOfDigi_Mean_Roll_vs_Sector_Wheel"<<i;
      if ( me = dbe_->get(globalFolder_ +"/"+ histoName.str()) ) {
	dbe_->removeElement(me->getName());
      }
      
      MULTWheel[i+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 12, 0.5, 12.5, 21, 0.5, 21.5);
      
      rpcUtils.labelXAxisSector( MULTWheel[i+2]);
      rpcUtils.labelYAxisRoll( MULTWheel[i+2], 0, i);
      
      histoName.str("");
      histoName<<"NumberOfDigi_Mean_Distribution_Wheel"<<i;
      if ( me = dbe_->get(globalFolder_ +"/"+ histoName.str()) ) {
	dbe_->removeElement(me->getName());
      }
      
      MULTDWheel[i+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 50.5);
    }//end wheels
    
    if (i == 0 || i< (-1 * numberOfDisks_) || i > numberOfDisks_)continue;

    int offset = numberOfDisks_;
    if (i>0) offset --; //used to skip case equale to zero
  
    histoName.str("");
    histoName<<"NumberOfDigi_Mean_Roll_vs_Sector_Disk"<<i;
    if ( me = dbe_->get(globalFolder_ +"/"+ histoName.str()) ) {
      dbe_->removeElement(me->getName());
    }
    MULTDisk[i+offset]   = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 6, 0.5, 6.5, 54, 0.5, 54.5);
    
    rpcUtils.labelXAxisSector(MULTDisk[i+offset] );
    rpcUtils.labelYAxisRoll(MULTDisk[i+offset], 1, i);

    histoName.str("");
    histoName<<"NumberOfDigi_Mean_Distribution_Disk"<<i;
    if ( me = dbe_->get(globalFolder_+"/"+ histoName.str()) ) {
      dbe_->removeElement(me->getName());
    }
    
    MULTDDisk[i+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 50.5);
  }//end loop on wheels and disks


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


void RPCMultiplicityTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {}

void RPCMultiplicityTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c){}

void RPCMultiplicityTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {
 // counts number of lumiSegs 
  if(lumiSeg.id().luminosityBlock()%prescaleFactor_ != 0) return; 

  edm::LogVerbatim ("multiplicity") <<"[RPCMultiplicityTest]: End of LS transition, performing the DQM client operation";

  //Clear Distributions
  int limit = numberOfDisks_ * 2;
  if(numberOfDisks_<2) limit = 5;

  for(int i =0 ; i<limit; i++){

    if(i < numberOfDisks_ * 2)
      MULTDDisk[i]->Reset();
    if(i<5)
      MULTDWheel[i]->Reset();
  }
 
  //Loop on MEs
  for (unsigned int  i = 0 ; i<myNumDigiMe_.size();i++){
    this->fillGlobalME(myDetIds_[i],myNumDigiMe_[i]);
  }//End loop on MEs
}
 
void RPCMultiplicityTest::endRun(const Run& r, const EventSetup& c){}

 void RPCMultiplicityTest::endJob(){}

void  RPCMultiplicityTest::fillGlobalME(RPCDetId & detId, MonitorElement * myMe){

  MonitorElement * MULT =NULL;
  MonitorElement * MULTD = NULL;

  if (detId.region()==0) {
    MULT = MULTWheel[detId.ring()+2];
    MULTD = MULTDWheel[detId.ring()+2];
  }else{
    if(((detId.station() * detId.region() ) + numberOfDisks_) >= 0 ){
    
      if(detId.region()<0){
      MULT = MULTDisk[(detId.station() * detId.region() ) + numberOfDisks_];
      MULTD = MULTDDisk[(detId.station() * detId.region() ) + numberOfDisks_];
      }else{
	MULT = MULTDisk[(detId.station() * detId.region() ) + numberOfDisks_ -1];
	MULTD = MULTDDisk[(detId.station() * detId.region() ) + numberOfDisks_-1];
      }
    }
  }


  if ( MULT && MULTD ){

  rpcdqm::utils rollNumber;
  int nr = rollNumber.detId2RollNr(detId);
  MULT->setBinContent(detId.sector(),nr, myMe->getMean() );

  MULTD->Fill(myMe->getMean());
  }

}

/* *  \author Anna Cimmino*/
#include "DQM/RPCMonitorDigi/interface/utils.h"
#include <DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h>
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <sstream>

RPCDeadChannelTest::RPCDeadChannelTest(const edm::ParameterSet& ps ){
 
  edm::LogVerbatim ("rpcdeadchanneltest") << "[RPCDeadChannelTest]: Constructor";

  useRollInfo_ = ps.getUntrackedParameter<bool>("UseRollInfo", false);

  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
}

RPCDeadChannelTest::~RPCDeadChannelTest(){}

void RPCDeadChannelTest::beginJob(std::string & workingFolder ){
 edm::LogVerbatim ("rpcdeadchanneltest") << "[RPCDeadChannelTest]: Begin Job";
  globalFolder_ =  workingFolder;

}


void RPCDeadChannelTest::getMonitorElements(std::vector<MonitorElement *> & meVector, std::vector<RPCDetId> & detIdVector, std::string & clientHistoName){

  for (unsigned int i = 0 ; i<meVector.size(); i++){
    
    std::string meName =  meVector[i]->getName();

    if(meName.find(clientHistoName) != std::string::npos){
      myOccupancyMe_.push_back(meVector[i]);
      myDetIds_.push_back(detIdVector[i]);
    }  
  }
}


void RPCDeadChannelTest::clientOperation(){
 
  edm::LogVerbatim ("rpcdeadchanneltest") <<"[RPCDeadChannelTest]:Client Operation";


  MonitorElement * DEAD = NULL;
 
  //Loop on chambers
    for (unsigned int  i = 0 ; i<myOccupancyMe_.size();i++){
      
      RPCDetId & detId = myDetIds_[i];
      MonitorElement * myMe = myOccupancyMe_[i];

      if (! myMe ) continue;

      const QReport * theOccupancyQReport = myMe->getQReport("DeadChannel_0"); 
 
      float deadFraction = 0.0 ;

      if(theOccupancyQReport) {
	
	float qtresult = theOccupancyQReport->getQTresult();
	// std::vector<dqm::me_util::Channel> badChannels = theOccupancyQReport->getBadChannels();
	deadFraction = 1.0 - qtresult;

      }else{
	int xBins = myMe->getNbinsX();
	float emptyBins = 0.0;
	for(int x = 1 ; x<= xBins ; x++){if(myMe->getBinContent(x) == 0 ) {emptyBins++;}}
	if (xBins != 0){	deadFraction = emptyBins/xBins;}
      }
     
     if (detId.region()==0)   DEAD = DEADWheel[detId.ring() + 2] ;
     else{
       if(-detId.station()+ numberOfDisks_ >= 0 ){
	 
	 if(detId.region()<0){
	   DEAD  = DEADDisk[-detId.station() + numberOfDisks_];
	 }else{
	   DEAD = DEADDisk[detId.station() + numberOfDisks_-1];
	 }
       }
     }

     if (DEAD){
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
       DEAD->setBinContent(xBin,yBin, deadFraction );

     }

    }//End loop on rolls in given chambers

}
 
void RPCDeadChannelTest::myBooker(DQMStore::IBooker & ibooker){

  ibooker.setCurrentFolder( globalFolder_);
  
  std::stringstream histoName;
  
  rpcdqm::utils rpcUtils;
  
  int limit = numberOfDisks_;
  if(numberOfDisks_ < 2) limit = 2;
  
  for (int i = -1 * limit; i<= limit;i++ ){//loop on wheels and disks
    if (i>-3 && i<3){//wheels
      histoName.str("");
      histoName<<"DeadChannelFraction_Roll_vs_Sector_Wheel"<<i;
      DEADWheel[i+2] = ibooker.book2D(histoName.str().c_str(), histoName.str().c_str(), 12, 0.5, 12.5, 21, 0.5, 21.5);
      
      for (int x = 1; x<=12; x++){
	for(int y=1; y<=21; y++){
	  DEADWheel[i+2]->setBinContent(x,y,-1);
	}
      }

      rpcUtils.labelXAxisSector( DEADWheel[i+2]);
      rpcUtils.labelYAxisRoll( DEADWheel[i+2], 0, i, useRollInfo_);
    }//end wheels
    
    if (i == 0  || i > numberOfDisks_ || i< (-1 * numberOfDisks_)){continue;}
    
    int offset = numberOfDisks_;
    if (i>0) {offset --;} //used to skip case equale to zero
    
    histoName.str("");
    histoName<<"DeadChannelFraction_Ring_vs_Segment_Disk"<<i;
    DEADDisk[i+offset] = ibooker.book2D(histoName.str().c_str(), histoName.str().c_str(),36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5);
    
    rpcUtils.labelXAxisSegment(DEADDisk[i+offset]);
    rpcUtils.labelYAxisRing(DEADDisk[i+offset], numberOfRings_ ,useRollInfo_);
        
  }//end loop on wheels and disks
  
  
}



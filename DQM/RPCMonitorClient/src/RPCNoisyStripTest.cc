#include <DQM/RPCMonitorClient/interface/RPCNoisyStripTest.h>
#include "DQM/RPCMonitorDigi/interface/utils.h"

//DQM Services
//#include "DQMServices/Core/interface/DQMStore.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include <FWCore/Framework/interface/ESHandle.h>

//Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"


RPCNoisyStripTest::RPCNoisyStripTest(const edm::ParameterSet& ps ){
  edm::LogVerbatim ("rpcnoisetest") << "[RPCNoisyStripTest]: Constructor";
 
  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
  useRollInfo_ = ps.getUntrackedParameter<bool>("UseRollInfo", false);
  testMode_ = ps.getUntrackedParameter<bool>("testMode", false);

}

RPCNoisyStripTest::~RPCNoisyStripTest(){}

void RPCNoisyStripTest::beginJob(std::string  & workingFolder){
 edm::LogVerbatim ("rpcnoisetest") << "[RPCNoisyStripTest]: Begin job ";
  globalFolder_ = workingFolder;
}


void RPCNoisyStripTest::clientOperation() {  

  edm::LogVerbatim ("rpcnoisetest") <<"[RPCNoisyStripTest]: Client Operation";
    
 //Loop on MEs
  for (unsigned int  i = 0 ; i<myOccupancyMe_.size();i++){
    this->fillGlobalME(myDetIds_[i],myOccupancyMe_[i]);
  }//End loop on MEs

}
 
void  RPCNoisyStripTest::myBooker(DQMStore::IBooker & ibooker){


 ibooker.setCurrentFolder( globalFolder_);

 std::stringstream histoName;

 rpcdqm::utils rpcUtils;

 for (int w = -2; w<= 2;w++ ){//loop on wheels and disks
  
   if(testMode_){
     histoName.str("");
     histoName<<"RPCNoisyStrips_Distribution_Wheel"<<w;     
     NOISEDWheel[w+2] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(),  6, -0.5, 5.5);
     
     
     histoName.str("");
     histoName<<"RPCStripsDeviation_Distribution_Wheel"<<w; 
     DEVDWheel[w+2] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(),  101, -0.01, 10.01);
   }

   histoName.str("");
   histoName<<"RPCNoisyStrips_Roll_vs_Sector_Wheel"<<w;
   NOISEWheel[w+2] = ibooker.book2D(histoName.str().c_str(), histoName.str().c_str() , 12, 0.5, 12.5, 21, 0.5, 21.5);
   rpcUtils.labelXAxisSector(NOISEWheel[w+2]);
   rpcUtils.labelYAxisRoll(NOISEWheel[w+2], 0, w, useRollInfo_);
 }

 
 
 for(int d = -numberOfDisks_; d<=numberOfDisks_; d++ ){//ENDCAP

   if (d == 0) continue; 

   int offset = numberOfDisks_;
   if (d>0) offset --;

   if (testMode_){
     histoName.str("");
     histoName<<"RPCNoisyStrips_Distribution_Disk"<<d;      
     NOISEDDisk[d+offset] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(),  6, -0.5, 5.5);
     
     
     histoName.str("");
     histoName<<"RPCStripsDeviation_Distribution_Disk"<<d;  
     DEVDDisk[d+offset] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(),  101, -0.01, 10.01);
   }

   histoName.str("");
   histoName<<"RPCNoisyStrips_Ring_vs_Segment_Disk"<<d;
   NOISEDisk[d+offset] = ibooker.book2D(histoName.str().c_str(), histoName.str().c_str() , 36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5);
   rpcUtils.labelXAxisSegment(NOISEDisk[d+offset]);
   rpcUtils.labelYAxisRing(NOISEDisk[d+offset], numberOfRings_, useRollInfo_);

 }
   
}


void  RPCNoisyStripTest::getMonitorElements(std::vector<MonitorElement *> & meVector, std::vector<RPCDetId> & detIdVector, std::string & clientHistoName){

 //Get NumberOfDigi ME for each roll
 for (unsigned int i = 0 ; i<meVector.size(); i++){
      
    std::string meName =  meVector[i]->getName();

    if(meName.find(clientHistoName) != std::string::npos){

     myOccupancyMe_.push_back(meVector[i]);
     myDetIds_.push_back(detIdVector[i]);
   }
 }

}

void  RPCNoisyStripTest::fillGlobalME(RPCDetId & detId, MonitorElement * myMe){

    std::stringstream meName;
    
    MonitorElement *  NOISE=NULL;
    MonitorElement * DEVD=NULL;
    MonitorElement * NOISED=NULL;

    if (detId.region()==0) { //BARREL
      NOISE = NOISEWheel[detId.ring()+2];
      if(testMode_) {
	DEVD = DEVDWheel[detId.ring()+2];
	NOISED= NOISEDWheel[detId.ring()+2];
      }
    }else if(detId.region()<0 && (-detId.station() + numberOfDisks_) >= 0 ){//ENDCAP-
      NOISE = NOISEDisk[ -detId.station() + numberOfDisks_];
      if(testMode_) {
	DEVD = DEVDDisk[ -detId.station()  + numberOfDisks_];
	NOISED= NOISEDDisk[-detId.station() + numberOfDisks_];
      }
    }else if((-detId.station() + numberOfDisks_)>= 0 ){//ENDCAP +
      NOISE = NOISEDisk[detId.station() + numberOfDisks_-1];
      if(testMode_) {
	DEVD = DEVDDisk[detId.station()  + numberOfDisks_-1];
	NOISED= NOISEDDisk[detId.station() + numberOfDisks_-1];
      }
    }
    
    
    int entries = (int) myMe -> getEntries();
    int bins = (int) myMe ->getNbinsX();
      
    std::vector<float> myvector;
	
    // count alive strips and alive strip values put in the vector
    for(int xbin =1 ; xbin <= bins ; xbin++) {	  
      float binContent = myMe->getBinContent(xbin);  
      if (binContent > 0) myvector.push_back(binContent);
    }
		
    
    int noisyStrips=0;
    // calculate mean on YAxis and check diff between bins and mean
    if (myvector.size()>0) {
      float ymean = entries/myvector.size(); //mean on Yaxis
      for(unsigned int i=0; i<myvector.size(); i++) {
	float deviation = myvector[i]/ymean;
	if(deviation > 3.5)  noisyStrips++;
	if(deviation > 5) deviation = 5; //overflow 
	if(DEVD) DEVD-> Fill(deviation);
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
      
      if(NOISE)  NOISE->setBinContent(xBin,yBin,noisyStrips); 
      if(NOISED) NOISED ->Fill(noisyStrips);
      
    }

}


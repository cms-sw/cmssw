#include <DQM/RPCMonitorClient/interface/RPCNoisyStripTest.h>
#include "DQM/RPCMonitorDigi/interface/utils.h"

//DQM Services
#include "DQMServices/Core/interface/DQMStore.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <FWCore/Framework/interface/ESHandle.h>

//Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"


RPCNoisyStripTest::RPCNoisyStripTest(const edm::ParameterSet& ps ){
  edm::LogVerbatim ("rpcnoisetest") << "[RPCNoisyStripTest]: Constructor";
 
  globalFolder_ = ps.getUntrackedParameter<std::string>("RPCGlobalFolder", "RPC/RecHits/SummaryHistograms");
  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 3);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
 
}

RPCNoisyStripTest::~RPCNoisyStripTest(){dbe_=0;}

void RPCNoisyStripTest::beginJob(DQMStore * dbe){
 edm::LogVerbatim ("rpcnoisetest") << "[RPCNoisyStripTest]: Begin job ";
 dbe_ = dbe;

}

void RPCNoisyStripTest::endRun(const edm::Run& r, const edm::EventSetup& iSetup,std::vector<MonitorElement *> meVector, std::vector<RPCDetId> detIdVector){
 edm::LogVerbatim ("rpcnoisetest") << "[RPCNoisyStripTest]: End run";
 
 
 MonitorElement* me;
 dbe_->setCurrentFolder( globalFolder_);

 std::stringstream histoName;

 rpcdqm::utils rpcUtils;

 int limit = numberOfDisks_;
 if(numberOfDisks_ < 2) limit = 2;
 for (int w = -1 * limit; w<= limit;w++ ){//loop on wheels and disks
   if (w>-3 && w<3){//Barrel
     histoName.str("");
     histoName<<"RPCNoisyStrips_Distribution_Wheel"<<w;     
     me =0;
     me = dbe_->get( globalFolder_ +"/"+ histoName.str());
     if ( 0!=me ) {
       dbe_->removeElement(me->getName());
     }
     NOISEDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  6, -0.5, 5.5);
     
     
     histoName.str("");
     histoName<<"RPCStripsDeviation_Distribution_Wheel"<<w; 
     me =0;
     me = dbe_->get( globalFolder_ +"/"+ histoName.str());
     if ( 0!=me ) {
       dbe_->removeElement(me->getName());
     }
     DEVDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  101, -0.01, 10.01);
     
     histoName.str("");
     histoName<<"RPCNoisyStrips_Roll_vs_Sector_Wheel"<<w;
     me =0;
     me = dbe_->get( globalFolder_ +"/"+ histoName.str());
     if ( 0!=me ) {
       dbe_->removeElement(me->getName());
     }
     
     NOISEWheel[w+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str() , 12, 0.5, 12.5, 21, 0.5, 21.5);
     rpcUtils.labelXAxisSector(NOISEWheel[w+2]);
     rpcUtils.labelYAxisRoll(NOISEWheel[w+2], 0, w);
   }

   if (w == 0  || w > numberOfDisks_ || w< (-1 * numberOfDisks_))continue;//ENDCAP

   int offset = numberOfDisks_;
   if (w>0) offset --; //used to skip case equale to zero

   histoName.str("");
   histoName<<"RPCNoisyStrips_Distribution_Disk"<<w;      
   me =0;
   me = dbe_->get( globalFolder_ +"/"+ histoName.str());
   if ( 0!=me ) {
     dbe_->removeElement(me->getName());
   }
   NOISEDDisk[w+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  6, -0.5, 5.5);
   
   
   histoName.str("");
   histoName<<"RPCStripsDeviation_Distribution_Disk"<<w;  
   me =0;
   me = dbe_->get( globalFolder_ +"/"+ histoName.str());
   if ( 0!=me ) {
     dbe_->removeElement(me->getName());
   }
   DEVDDisk[w+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  101, -0.01, 10.01);
   
   histoName.str("");
   histoName<<"RPCNoisyStrips_Ring_vs_Segment_Disk"<<w;
   me =0;
   me = dbe_->get( globalFolder_ +"/"+ histoName.str());
   if ( 0!=me ) {
     dbe_->removeElement(me->getName());
   }
   
   NOISEDisk[w+offset] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str() , 36, 0.5, 36.5, 3*numberOfRings_, 0.5,3*numberOfRings_+ 0.5);
   rpcUtils.labelXAxisSegment(NOISEDisk[w+offset]);
   rpcUtils.labelYAxisRing(NOISEDisk[w+offset], numberOfRings_);

 }//end loop wheel and disks
   
 //Get NumberOfDigi ME for each roll
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

void RPCNoisyStripTest::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context){} 

void RPCNoisyStripTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {}

void RPCNoisyStripTest::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {}

void RPCNoisyStripTest::clientOperation(edm::EventSetup const& iSetup) {  


  edm::LogVerbatim ("rpcnoisetest") <<"[RPCNoisyStripTest]: Client Operation";
  
  //Clear Distributions
  int limit = numberOfDisks_ * 2;
  if(numberOfDisks_<2) limit = 5;

  for(int i =0 ; i<limit; i++){

    if(i < numberOfDisks_ * 2){
      DEVDDisk[i]->Reset();
      NOISEDDisk[i]->Reset();
    }
    if(i<5){
      DEVDWheel[i]->Reset();
      NOISEDWheel[i]->Reset();
    }
  }
  
 //Loop on MEs
  for (unsigned int  i = 0 ; i<myOccupancyMe_.size();i++){
    this->fillGlobalME(myDetIds_[i],myOccupancyMe_[i], iSetup);
  }//End loop on MEs
}


 
void  RPCNoisyStripTest::beginRun(const edm::Run& r, const edm::EventSetup& c){}

void  RPCNoisyStripTest::endJob(){}

void  RPCNoisyStripTest::fillGlobalME(RPCDetId & detId, MonitorElement * myMe,edm::EventSetup const& iSetup){

 //   ESHandle<RPCGeometry> rpcgeo;
//     iSetup.get<MuonGeometryRecord>().get(rpcgeo);
 
    std::stringstream meName;
    
    MonitorElement *  NOISE=NULL;
    MonitorElement * DEVD=NULL;
    MonitorElement * NOISED=NULL;

    if (detId.region()==0) { //BARREL
      NOISE = NOISEWheel[detId.ring()+2];
      DEVD = DEVDWheel[detId.ring()+2];
      NOISED= NOISEDWheel[detId.ring()+2];
    }else if(detId.region()<0 && (-detId.station() + numberOfDisks_) >= 0 ){//ENDCAP-
      NOISE = NOISEDisk[ -detId.station() + numberOfDisks_];
      DEVD = DEVDDisk[ -detId.station()  + numberOfDisks_];
      NOISED= NOISEDDisk[-detId.station() + numberOfDisks_];
    }else if((-detId.station() + numberOfDisks_)>= 0 ){//ENDCAP +
      NOISE = NOISEDisk[detId.station() + numberOfDisks_-1];
      DEVD = DEVDDisk[detId.station()  + numberOfDisks_-1];
      NOISED= NOISEDDisk[detId.station() + numberOfDisks_-1];
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
	
      if(NOISE && NOISED ){
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
	
	NOISE->setBinContent(xBin,yBin,noisyStrips); 
	NOISED ->Fill(noisyStrips);
      }
    }

}


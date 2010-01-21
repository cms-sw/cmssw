/*  \author Anna Cimmino*/
#include <cmath>
#include <sstream>
#include <DQM/RPCMonitorClient/interface/RPCOccupancyTest.h>
#include "DQM/RPCMonitorDigi/interface/utils.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
RPCOccupancyTest::RPCOccupancyTest(const ParameterSet& ps ){
  LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: Constructor";
  
  globalFolder_ = ps.getUntrackedParameter<string>("RPCGlobalFolder", "RPC/RecHits/SummaryHistograms");
  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 3);
}

RPCOccupancyTest::~RPCOccupancyTest(){
  dbe_=0;
}

void RPCOccupancyTest::beginJob(DQMStore * dbe){
 LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: Begin job ";
 dbe_=dbe;
}

void RPCOccupancyTest::endRun(const Run& r, const EventSetup& c,vector<MonitorElement *> meVector, vector<RPCDetId> detIdVector){
 LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: End run";
 
 
 MonitorElement* me;
 dbe_->setCurrentFolder( globalFolder_);

 stringstream histoName;
 rpcdqm::utils rpcUtils;

 int limit = numberOfDisks_;
 if(numberOfDisks_ < 2) limit = 2;
  
  for (int w = -1 *limit; w<=limit; w++ ){//loop on wheels and disks
    if (w>-3 && w<3){//Barrel
      histoName.str("");
      histoName<<"AsymmetryLeftRight_Roll_vs_Sector_Wheel"<<w;
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      
      AsyMeWheel[w+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
   
      rpcUtils.labelXAxisSector(  AsyMeWheel[w+2]);
      rpcUtils.labelYAxisRoll( AsyMeWheel[w+2], 0, w);
      
      histoName.str("");
      histoName<<"AsymmetryLeftRight_Distribution_Wheel"<<w;  
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      AsyMeDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, -0.1, 1.1);
      
       
      histoName.str("");
      histoName<<"OccupancyNormByGeoAndRPCEvents_Wheel"<<w;
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      
      NormOccupWheel[w+2] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
      
      rpcUtils.labelXAxisSector(  NormOccupWheel[w+2]);
      rpcUtils.labelYAxisRoll(  NormOccupWheel[w+2], 0, w);
   
      histoName.str("");
      histoName<<"OccupancyNormByGeoAndRPCEvents_Distribution_Wheel"<<w;   
      me = 0;
      me = dbe_->get( globalFolder_+"/"+ histoName.str());
      if ( 0!=me  ) {
	dbe_->removeElement(me->getName());
      }
      NormOccupDWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.0, 0.205);
    }//end Barrel

    if (w == 0 || w< (-1 * numberOfDisks_) || w > numberOfDisks_)continue;
    
    int offset = numberOfDisks_;
    if (w>0) offset --; //used to skip case equale to zero
    
    histoName.str("");
    histoName<<"AsymmetryLeftRight_Roll_vs_Sector_Disk"<<w;
    me = 0;
    me = dbe_->get( globalFolder_+"/"+ histoName.str());
    if ( 0!=me  ) {
      dbe_->removeElement(me->getName());
    }
      
    AsyMeDisk[w+offset] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  6, 0.5, 6.5, 54, 0.5, 54.5);
    
    rpcUtils.labelXAxisSector(AsyMeDisk[w+offset]);
    rpcUtils.labelYAxisRoll(AsyMeDisk[w+offset], 1, w);
    
    histoName.str("");
    histoName<<"AsymmetryLeftRight_Distribution_Disk"<<w;      
    me = 0;
    me = dbe_->get( globalFolder_+"/"+ histoName.str());
    if ( 0!=me  ) {
       dbe_->removeElement(me->getName());
    }
    AsyMeDDisk[w+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, -0.1, 1.1);
    
    
    histoName.str("");
    histoName<<"OccupancyNormByGeoAndRPCEvents_Disk"<<w;
    me = 0;
    me = dbe_->get( globalFolder_+"/"+ histoName.str());
    if ( 0!=me  ) {
      dbe_->removeElement(me->getName());
    }
    
    NormOccupDisk[w+offset] = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  6, 0.5, 6.5, 54, 0.5, 54.5);
    
    rpcUtils.labelXAxisSector(  NormOccupDisk[w+offset]);
    rpcUtils.labelYAxisRoll( NormOccupDisk[w+offset], 1, w);
    
    histoName.str("");
    histoName<<"OccupancyNormByGeoAndRPCEvents_Distribution_Disk"<<w;  
    me = 0;
    me = dbe_->get( globalFolder_+"/"+ histoName.str());
    if ( 0!=me  ) {
      dbe_->removeElement(me->getName());
    }
    NormOccupDDisk[w+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.0, 0.205);
    
  }
  
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

void RPCOccupancyTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCOccupancyTest::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCOccupancyTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {}

void RPCOccupancyTest::clientOperation(EventSetup const& iSetup) {
  LogVerbatim ("rpceventsummary") <<"[RPCOccupancyTest]: Client Operation";

   MonitorElement * RPCEvents = dbe_->get(globalFolder_ +"/RPCEvents");  
   rpcevents_ = RPCEvents -> getEntries(); 

   //Clear distributions

   //Clear Distributions
   int limit = numberOfDisks_ * 2;
   if(numberOfDisks_<2) limit = 5;
   for(int i =0 ; i<limit; i++){
     if(i < numberOfDisks_ * 2){
       AsyMeDDisk[i]->Reset();
       NormOccupDDisk[i]->Reset();
     }
     if(i<5){
      AsyMeDWheel[i]->Reset();
      NormOccupDWheel[i]->Reset();
     }
   }
   
 //Loop on MEs
  for (unsigned int  i = 0 ; i<myOccupancyMe_.size();i++){
    this->fillGlobalME(myDetIds_[i],myOccupancyMe_[i]);
  }//End loop on MEs
}

void RPCOccupancyTest::endJob(void) {}
void RPCOccupancyTest::beginRun(const Run& r, const EventSetup& c) {}


void RPCOccupancyTest::fillGlobalME(RPCDetId & detId, MonitorElement * myMe){
     if (!myMe) return;
    
    MonitorElement * AsyMe=NULL;      //Left Right Asymetry 
    MonitorElement * AsyMeD=NULL; 
    MonitorElement * NormOccup=NULL;
    MonitorElement * NormOccupD=NULL;
      
 
    if(detId.region() ==0){
      AsyMe= AsyMeWheel[detId.ring()+2];
      AsyMeD= AsyMeDWheel[detId.ring()+2];
      NormOccup=NormOccupWheel[detId.ring()+2];
      NormOccupD=NormOccupDWheel[detId.ring()+2];

    }else{

      if(((detId.station() * detId.region() ) + numberOfDisks_) >= 0 ){
	
	if(detId.region()<0){
	  AsyMe= AsyMeDisk[(detId.station() * detId.region() ) + numberOfDisks_];
	  AsyMeD= AsyMeDDisk[(detId.station() * detId.region() ) + numberOfDisks_];
	  NormOccup=NormOccupDisk[(detId.station() * detId.region() ) + numberOfDisks_];
	  NormOccupD=NormOccupDDisk[(detId.station() * detId.region() ) + numberOfDisks_];
	}else{
	  AsyMe= AsyMeDisk[(detId.station() * detId.region() ) + numberOfDisks_-1];
	  AsyMeD= AsyMeDDisk[(detId.station() * detId.region() ) + numberOfDisks_-1];
	  NormOccup=NormOccupDisk[(detId.station() * detId.region() ) + numberOfDisks_-1];
	  NormOccupD=NormOccupDDisk[(detId.station() * detId.region() ) + numberOfDisks_-1];
	}
      }


    }

    rpcdqm::utils prova;	  
    int nr = prova.detId2RollNr(detId);

                                              
	
    int stripInRoll=myMe->getNbinsX();
    float FOccupancy=0;
    float BOccupancy=0;
    
    float  totEnt =  myMe->getEntries();
    for(int strip = 1 ; strip<=stripInRoll; strip++){
      if(strip<=stripInRoll/2) FOccupancy+=myMe->getBinContent(strip);
      else  BOccupancy+=myMe->getBinContent(strip);
    }
	    
    float asym =  fabs((FOccupancy - BOccupancy )/totEnt);
    
    if(AsyMe)  AsyMe->setBinContent(detId.sector(),nr,asym );

    if(AsyMeD) AsyMeD->Fill(asym);
	
    float normoccup = 0;
    if(stripInRoll*rpcevents_ !=0)
      normoccup = totEnt/(stripInRoll*rpcevents_)*10;
    if(NormOccup)  NormOccup->setBinContent(detId.sector(),nr, normoccup);
    if(NormOccupD) NormOccupD->Fill(normoccup);
}






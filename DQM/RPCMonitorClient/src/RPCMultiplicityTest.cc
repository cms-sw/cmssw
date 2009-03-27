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
<<<<<<< RPCMultiplicityTest.cc
=======
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
>>>>>>> 1.5

#include <sstream>

using namespace edm;
using namespace std;

RPCMultiplicityTest::RPCMultiplicityTest(const ParameterSet& ps ){
  LogVerbatim ("multiplicity") << "[RPCMultiplicityTest]: Constructor";

<<<<<<< RPCMultiplicityTest.cc
  globalFolder_ = ps.getUntrackedParameter<string>("RPCGlobalFolder", "RPC/RecHits/SummaryHistograms");
  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 3);

=======
  globalFolder_ = ps.getUntrackedParameter<string>("GlobalFolder", "SummaryHistograms");
  prescaleFactor_ = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);
  prefixDir_= ps.getUntrackedParameter<string>("GlobalFolder", "RPC/RecHits");
>>>>>>> 1.5
}

RPCMultiplicityTest::~RPCMultiplicityTest(){
  dbe_ = 0;
}

<<<<<<< RPCMultiplicityTest.cc

void RPCMultiplicityTest::beginJob(DQMStore *  dbe ){
 LogVerbatim ("multiplicity") << "[RPCMultiplicityTest]: Begin job";
 dbe_=dbe;
}
=======
void RPCMultiplicityTest::beginJob(const EventSetup& iSetup){
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc

void RPCMultiplicityTest::beginRun(const Run& r, const EventSetup& iSetup,vector<MonitorElement *> meVector, vector<RPCDetId> detIdVector){
=======
 LogVerbatim ("deadChannel") << "[RPCMultiplicityTest]: Begin job";
 dbe_ = Service<DQMStore>().operator->();
 dbe_->setVerbose(0);
}
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
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
=======
void RPCMultiplicityTest::beginRun(const Run& r, const EventSetup& iSetup){

 edm::LogVerbatim ("deadChannel") << "[RPCMultiplicityTest]: Begin run";

 MonitorElement* me=NULL;
 dbe_->setCurrentFolder( prefixDir_+"/"+globalFolder_);

 stringstream histoName;

 for (int i = -4; i<=4;i++ ){//loop on wheels and disks
   if (i>-3 && i<3){//wheels
     histoName.str("");
     histoName<<"NumberOfDigi_Roll_vs_Sector_Wheel"<<i;
     if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
     }
     
     me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 12, 0.5, 12.5, 21, 0.5, 21.5);
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
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
=======
     //set x axis labels
     for(int bin =1; bin< (me->getNbinsX() + 1);bin++) {
       histoName.str("");
       histoName<<"Sec"<<bin;
       me->setBinLabel(bin,histoName.str().c_str(),1);
     }

     histoName.str("");
     histoName<<"NumberOfDigi_Distribution_Wheel"<<i;
     if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
     }
     me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 50.5);
   
     
   } //end wheels
   
   if(i!=0){//Forward
     histoName.str("");
     histoName<<"NumberOfDigi_Roll_vs_Sector_Disk"<<i;
     if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
     }
     me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 6, 0.5, 6.5, 54, 0.5, 54.5);
   
   
   //set x axis labels
   for(int bin =1; bin< (me->getNbinsX() + 1);bin++) {
     histoName.str("");
     histoName<<"Sec"<<bin;
     me->setBinLabel(bin,histoName.str().c_str(),1);
   }
   
   histoName.str("");
   histoName<<"NumberOfDigi_Distribution_Disk"<<i;
   if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }
   
   me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 50.5);
   }
 }//end loop on wheels and disks
 
 //Get NumberOfDigi ME for each roll
 rpcdqmclient::clientTools tool;
 myNumDigiMe_ = tool.constructMEVector(iSetup, prefixDir_, "NumberOfDigi", dbe_);
 myDetIds_ = tool.getAssociatedRPCdetId();
>>>>>>> 1.5
}

<<<<<<< RPCMultiplicityTest.cc

=======
>>>>>>> 1.5
void RPCMultiplicityTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {}

void RPCMultiplicityTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c){}
<<<<<<< RPCMultiplicityTest.cc

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
=======

void RPCMultiplicityTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {
>>>>>>> 1.5
 
<<<<<<< RPCMultiplicityTest.cc
  //Loop on MEs
  for (unsigned int  i = 0 ; i<myNumDigiMe_.size();i++){
    this->fillGlobalME(myDetIds_[i],myNumDigiMe_[i]);
  }//End loop on MEs
}
 
void RPCMultiplicityTest::endRun(const Run& r, const EventSetup& c){}
=======

  //REMEMBER TO CLEAR DISTRIBUTIONS

>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
 void RPCMultiplicityTest::endJob(){}
=======
  edm::LogVerbatim ("deadChannel") <<"[RPCMultiplicityTest]: End of LS transition, performing the DQM client operation";
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
void  RPCMultiplicityTest::fillGlobalME(RPCDetId & detId, MonitorElement * myMe){
=======
  // counts number of lumiSegs 
  int nLumiSegs = lumiSeg.id().luminosityBlock();

  //check some statements and prescale Factor
  if(nLumiSegs%prescaleFactor_ == 0) {

    //Loop on MEs
    for (unsigned int  i = 0 ; i<myNumDigiMe_.size();i++){
      this->fillGlobalME(myDetIds_[i],myNumDigiMe_[i],iSetup);
    }//End loop on MEs
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
  MonitorElement * MULT =NULL;
  MonitorElement * MULTD = NULL;
=======
  }
}
 
void RPCMultiplicityTest::endRun(const Run& r, const EventSetup& c){}

void RPCMultiplicityTest::endJob(){}
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
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
=======
//
//User Defined methods
//
void  RPCMultiplicityTest::fillGlobalME(RPCDetId & detId, MonitorElement * myMe, EventSetup const& iSetup){
 
  edm::ESHandle<RPCGeometry> rpcgeo;
  iSetup.get<MuonGeometryRecord>().get(rpcgeo); 
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
=======
  
  MonitorElement * myGlobalMe;
  MonitorElement * distMe;
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
  if ( MULT && MULTD ){
=======
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
  rpcdqm::utils rollNumber;
  int nr = rollNumber.detId2RollNr(detId);
  MULT->setBinContent(detId.sector(),nr, myMe->getMean() );
=======
  stringstream meName, distName;
  
   
  if (detId.region()==0) {
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
  MULTD->Fill(myMe->getMean());
=======
    meName.str("");
    meName<<prefixDir_+"/"+ globalFolder_+"/NumberOfDigi_Roll_vs_Sector_Wheel"<<detId.ring();
    distName.str("");
    distName<<prefixDir_+"/"+ globalFolder_+"/NumberOfDigi_Distribution_Wheel"<<detId.ring();
  

  }else{
    meName.str("");
    meName<<prefixDir_+"/"+ globalFolder_+"/NumberOfDigi_Roll_vs_Sector_Disk"<<detId.region()*detId.station();
    distName.str("");
    distName<<prefixDir_+"/"+ globalFolder_+"/NumberOfDigi_Distribution_Disk"<<detId.ring();
>>>>>>> 1.5
  }
<<<<<<< RPCMultiplicityTest.cc
=======
  myGlobalMe = dbe_->get(meName.str());
  distMe =   dbe_->get(distName.str());
>>>>>>> 1.5

<<<<<<< RPCMultiplicityTest.cc
=======
  if (myGlobalMe && distMe){

  rpcdqm::utils rollNumber;
  int nr = rollNumber.detId2RollNr(detId);
  myGlobalMe->setBinContent(detId.sector(),nr, myMe->getMean() );
  
  distMe->Fill(myMe->getMean());

  RPCGeomServ RPCname(detId);	  
  string YLabel = RPCname.shortname();
  myGlobalMe->setBinLabel(nr, YLabel, 2);
  }
>>>>>>> 1.5
}
<<<<<<< RPCMultiplicityTest.cc
=======


>>>>>>> 1.5

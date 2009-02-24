/*
 *  \author Anna Cimmino
 */
#include "DQM/RPCMonitorDigi/interface/utils.h"
#include <DQM/RPCMonitorClient/interface/RPCMultiplicityTest.h>
#include <DQM/RPCMonitorClient/interface/clientTools.h>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DataFormats
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

// Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <sstream>

using namespace edm;
using namespace std;

RPCMultiplicityTest::RPCMultiplicityTest(const ParameterSet& ps ){
 
  LogVerbatim ("deadChannel") << "[RPCMultiplicityTest]: Constructor";

  globalFolder_ = ps.getUntrackedParameter<string>("GlobalFolder", "SummaryHistograms");
  prescaleFactor_ = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);
  prefixDir_= ps.getUntrackedParameter<string>("GlobalFolder", "RPC/RecHits");
}

RPCMultiplicityTest::~RPCMultiplicityTest(){
  dbe_ = 0;
}

void RPCMultiplicityTest::beginJob(const EventSetup& iSetup){

 LogVerbatim ("deadChannel") << "[RPCMultiplicityTest]: Begin job";
 dbe_ = Service<DQMStore>().operator->();
 dbe_->setVerbose(0);
}

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
}

void RPCMultiplicityTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {}

void RPCMultiplicityTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c){}

void RPCMultiplicityTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {
 

  //REMEMBER TO CLEAR DISTRIBUTIONS


  edm::LogVerbatim ("deadChannel") <<"[RPCMultiplicityTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  int nLumiSegs = lumiSeg.id().luminosityBlock();

  //check some statements and prescale Factor
  if(nLumiSegs%prescaleFactor_ == 0) {

    //Loop on MEs
    for (unsigned int  i = 0 ; i<myNumDigiMe_.size();i++){
      this->fillGlobalME(myDetIds_[i],myNumDigiMe_[i],iSetup);
    }//End loop on MEs

  }
}
 
void RPCMultiplicityTest::endRun(const Run& r, const EventSetup& c){}

void RPCMultiplicityTest::endJob(){}

//
//User Defined methods
//
void  RPCMultiplicityTest::fillGlobalME(RPCDetId & detId, MonitorElement * myMe, EventSetup const& iSetup){
 
  edm::ESHandle<RPCGeometry> rpcgeo;
  iSetup.get<MuonGeometryRecord>().get(rpcgeo); 

  
  MonitorElement * myGlobalMe;
  MonitorElement * distMe;


  stringstream meName, distName;
  
   
  if (detId.region()==0) {

    meName.str("");
    meName<<prefixDir_+"/"+ globalFolder_+"/NumberOfDigi_Roll_vs_Sector_Wheel"<<detId.ring();
    distName.str("");
    distName<<prefixDir_+"/"+ globalFolder_+"/NumberOfDigi_Distribution_Wheel"<<detId.ring();
  

  }else{
    meName.str("");
    meName<<prefixDir_+"/"+ globalFolder_+"/NumberOfDigi_Roll_vs_Sector_Disk"<<detId.region()*detId.station();
    distName.str("");
    distName<<prefixDir_+"/"+ globalFolder_+"/NumberOfDigi_Distribution_Disk"<<detId.ring();
  }
  myGlobalMe = dbe_->get(meName.str());
  distMe =   dbe_->get(distName.str());

  if (myGlobalMe && distMe){

  rpcdqm::utils rollNumber;
  int nr = rollNumber.detId2RollNr(detId);
  myGlobalMe->setBinContent(detId.sector(),nr, myMe->getMean() );
  
  distMe->Fill(myMe->getMean());

  RPCGeomServ RPCname(detId);	  
  string YLabel = RPCname.shortname();
  myGlobalMe->setBinLabel(nr, YLabel, 2);
  }
}



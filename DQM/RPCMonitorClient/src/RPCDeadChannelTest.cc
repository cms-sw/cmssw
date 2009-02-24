/*
 *  \author Anna Cimmino
 */
#include "DQM/RPCMonitorDigi/interface/utils.h"
#include <DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h>
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"
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

RPCDeadChannelTest::RPCDeadChannelTest(const ParameterSet& ps ){
 
  LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: Constructor";

  globalFolder_ = ps.getUntrackedParameter<string>("GlobalFolder", "SummaryHistograms");
  prescaleFactor_ = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);
  prefixDir_= ps.getUntrackedParameter<string>("GlobalFolder", "RPC/RecHits");
}

RPCDeadChannelTest::~RPCDeadChannelTest(){
  dbe_ = 0;
}

void RPCDeadChannelTest::beginJob(const EventSetup& iSetup){

 LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: Begin job";
 dbe_ = Service<DQMStore>().operator->();
 dbe_->setVerbose(0);
}

void RPCDeadChannelTest::beginRun(const Run& r, const EventSetup& iSetup){

 edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: Begin run";

 MonitorElement* me;
 dbe_->setCurrentFolder( prefixDir_+"/"+globalFolder_);
 
 stringstream histoName;

 for (int i = -4; i<=4;i++ ){//loop on wheels and disks
   if (i>-3 && i<3){//wheels
     histoName.str("");
     histoName<<"DeadChannelFraction_Roll_vs_Sector_Wheel"<<i;
     if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
     }
     
     me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 12, 0.5, 12.5, 21, 0.5, 21.5);
     
     for(int bin =1; bin<13;bin++) {
       histoName.str("");
       histoName<<"Sec"<<bin;
       me->setBinLabel(bin,histoName.str().c_str(),1);
     }

     histoName.str("");
     histoName<<"ClusterSize_AliveStrips_Roll_vs_Sector_Wheel"<<i;
     if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
     }
     
     me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 12, 0.5, 12.5, 21, 0.5, 21.5);

     for(int bin =1; bin<13;bin++) {
       histoName.str("");
       histoName<<"Sec"<<bin;
       me->setBinLabel(bin,histoName.str().c_str(),1);
     }
 histoName.str("");
     histoName<<"ClusterSize_AliveStrips_Distribution_Wheel"<<i;
     if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
     }
     
     me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),40, 0.5, 20.5 );



   }//end wheels

   if (i ==0 )continue;

   histoName.str("");
   histoName<<"DeadChannelFraction_Roll_vs_Sector_Disk"<<i;
   if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }
   me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 6, 0.5, 6.5, 54, 0.5, 54.5);
   
   for(int bin =1; bin<7;bin++) {
     histoName.str("");
     histoName<<"Sec"<<bin;
     me->setBinLabel(bin,histoName.str().c_str(),1);
   }

   histoName.str("");
   histoName<<"ClusterSize_AliveStrips_Roll_vs_Sector__Disk"<<i;
   if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }
   me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(), 6, 0.5, 6.5, 54, 0.5, 54.5);
   
   for(int bin =1; bin<7;bin++) {
     histoName.str("");
     histoName<<"Sec"<<bin;
     me->setBinLabel(bin,histoName.str().c_str(),1);
   }
 histoName.str("");
 histoName<<"ClusterSize_AliveStrips_Distribution_Disk"<<i;
     if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
     }
     
     me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),40, 0.5, 20.5 );




 }//end loop on wheels and disks

 //Start booking global histos
 histoName.str("");
 histoName<<"DeadChannelPercentage_Barrel";
 if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
 }
 me = dbe_->book2D(histoName.str().c_str(), "Dead Channel Fraction in Barrel", 12, 0.5, 12.5, 5, -2.5, 2.5);

 for(int bin =1; bin<13; bin++) {//set labels
   histoName.str("");
   histoName<<"Sec"<< bin;
   me->setBinLabel( bin,histoName.str().c_str(),1);
   if(bin<5) {
     histoName.str("");
     histoName<<"Wheel"<<bin-3;
     me->setBinLabel(bin,histoName.str().c_str(),2);
   }
 }
 
 histoName.str("");
 histoName<<"DeadChannelPercentage_EndcapPositive";
 if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
   dbe_->removeElement(me->getName());
 }
 me = dbe_->book2D(histoName.str().c_str(), "Dead Channel Fraction in Endcap+", 6, 0.5, 6.5, 4, -2, 2);

 for(int bin =1; bin<7; bin++) {//set labels
   histoName.str("");
   histoName<<"Sec"<< bin;
   me->setBinLabel( bin,histoName.str().c_str(),1);
   if(bin<5) {
     histoName.str("");
     histoName<<"Disk"<<bin;
     me->setBinLabel(bin,histoName.str().c_str(),2);
   }
 }
 
 histoName.str("");
 histoName<<"DeadChannelPercentage_EndcapNegative";
 if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
   dbe_->removeElement(me->getName());
 }
 me = dbe_->book2D(histoName.str().c_str(), "Dead Channel Fraction in Endcap-", 6, 0.5, 6.5,4, -2, 2); 

 for(int bin =1; bin<7; bin++) {//set labels
   histoName.str("");
   histoName<<"Sec"<< bin;
   me->setBinLabel( bin,histoName.str().c_str(),1);
   if(bin<5) {
     histoName.str("");
     histoName<<"Disk"<<bin;
     me->setBinLabel(bin,histoName.str().c_str(),2);
   }
 }

 ESHandle<RPCGeometry> rpcGeo;
 iSetup.get<MuonGeometryRecord>().get(rpcGeo);
 
 //loop on all geometry and get all histos
 for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
   if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
     RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
     std::vector< const RPCRoll*> roles = (ch->rolls());
     //Loop on rolls in given chamber
     for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
       RPCDetId detId = (*r)->id();
      
       //Get Occupancy ME for roll
       RPCGeomServ RPCname(detId);	   
       
       RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
       MonitorElement * myMe = dbe_->get(prefixDir_+"/"+ folderStr->folderStructure(detId)+"/Occupancy_"+RPCname.name()); 
       if (!myMe)continue;

       myOccupancyMe_.push_back(myMe);
       myDetIds_.push_back(detId);
       myRollNames_.push_back(RPCname.name());
     }
   }
 }//end loop on all geometry and get all histos
}

void RPCDeadChannelTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {}

void RPCDeadChannelTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c){}

void RPCDeadChannelTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {
 
  edm::LogVerbatim ("deadChannel") <<"[RPCDeadChannelTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  int nLumiSegs = lumiSeg.id().luminosityBlock();

  //check some statements and prescale Factor
  if(nLumiSegs%prescaleFactor_ == 0) {

    //Loop on chambers
    for (unsigned int  i = 0 ; i<myOccupancyMe_.size();i++){
      this->CalculateDeadChannelPercentage(myDetIds_[i],myOccupancyMe_[i],iSetup);
    }//End loop on rolls in given chambers

    this->fillDeadChannelHisto(barrelMap_, 0);
    this->fillDeadChannelHisto(endcapMap_, 1);
    // this->fillDeadChannelHisto(endcapMap_, -1);
  }
}
 
void RPCDeadChannelTest::endRun(const Run& r, const EventSetup& c){}

void RPCDeadChannelTest::endJob(){}

//
//User Defined methods
//
void  RPCDeadChannelTest::CalculateDeadChannelPercentage(RPCDetId & detId, MonitorElement * myMe, EventSetup const& iSetup){
 
  edm::ESHandle<RPCGeometry> rpcgeo;
  iSetup.get<MuonGeometryRecord>().get(rpcgeo); 

  const RPCRoll * rpcRoll = rpcgeo->roll(detId);      

  unsigned int nstrips =rpcRoll->nstrips();
  
  MonitorElement * myGlobalMe;
  MonitorElement * myGlobalMe2;
  
   stringstream meName;

  const QReport * theOccupancyQReport = myMe->getQReport("DeadChannel_0");  
  if(theOccupancyQReport) {
  
  vector<dqm::me_util::Channel> badChannels = theOccupancyQReport->getBadChannels();
  
  if (detId.region()==0) {
    barrelMap_[detId.ring()][detId.sector()].first += badChannels.size();
    barrelMap_[detId.ring()][detId.sector()].second += nstrips ;
    meName.str("");
    meName<<prefixDir_+"/"+ globalFolder_+"/DeadChannelFraction_Roll_vs_Sector_Wheel"<<detId.ring();
  }else{
    endcapMap_[detId.region()*detId.station()][detId.sector()].first +=  badChannels.size();
    endcapMap_[detId.region()*detId.station()][detId.sector()].second+=nstrips;
    meName.str("");
    meName<<prefixDir_+"/"+ globalFolder_+"/DeadChannelsFractionRoll_vs_Sector_Disk"<<detId.region()*detId.station();
  }
  myGlobalMe = dbe_->get(meName.str());
  
  if (myGlobalMe){

  rpcdqm::utils rollNumber;
  int nr = rollNumber.detId2RollNr(detId);
  float badchanfrac = badChannels.size()*100/nstrips;
  if(badchanfrac==0) badchanfrac = 0.001;
  myGlobalMe->setBinContent(detId.sector(),nr, badchanfrac );
  
  RPCGeomServ RPCname(detId);	  
  string YLabel = RPCname.shortname();
  myGlobalMe->setBinLabel(nr, YLabel, 2);
  
  meName.str("");
  meName<<prefixDir_+"/"+ globalFolder_+"/ClusterSize_AliveStrips_Roll_vs_Sector_Wheel"<<detId.ring();
  myGlobalMe = dbe_->get(meName.str());

  meName.str("");
  meName<<prefixDir_+"/"+ globalFolder_+"/ClusterSizeMean_Roll_vs_Sector_Wheel"<<detId.ring();	
  myGlobalMe2 = dbe_->get(meName.str());
     
  if ( myGlobalMe &&  myGlobalMe2){        	

  int goodCh =nstrips-badChannels.size();
  if(badChannels.size()<nstrips) myGlobalMe->setBinContent(detId.sector(),nr, myGlobalMe2->getBinContent(detId.sector(),nr)/goodCh );
  else  myGlobalMe->setBinContent(detId.sector(),nr, 1 ); 

  myGlobalMe->setBinLabel(nr,YLabel , 2);
  meName.str("");
  meName<<prefixDir_+"/"+ globalFolder_+"/ClusterSize_AliveStrips_Distribution_Wheel"<<detId.ring();
  myGlobalMe = dbe_->get(meName.str());
  
  if ( myGlobalMe) myGlobalMe->Fill( myGlobalMe2->getBinContent(detId.sector(),nr)/goodCh);

  }

}
  }
}



//Fill report summary
void  RPCDeadChannelTest::fillDeadChannelHisto(const map<int,map<int,pair<float,float> > > & sumMap, int region){

  MonitorElement *   regionME=NULL;

  map<int,map<int,pair<float,float> > >::const_iterator itr;
 
  if (sumMap.size()!=0){
    for (itr=sumMap.begin(); itr!=sumMap.end(); itr++){
      for (map< int ,  pair<float,float> >::const_iterator meItr = (*itr).second.begin(); meItr!=(*itr).second.end();meItr++){
	  if (region==0){
	    regionME = dbe_->get(prefixDir_+"/"+globalFolder_ +"/DeadChannelPercentage_Barrel");
	    regionME->setBinContent((*meItr).first, (*itr).first + 3,(*meItr).second.first/(*meItr).second.second );
	  }else {
	    regionME = dbe_->get(prefixDir_+"/"+globalFolder_ +"/DeadChannelPercentage_EndcapPositive");  
	    regionME->setBinContent((*meItr).first, (*itr).first ,(*meItr).second.first/(*meItr).second.second );

	    regionME = dbe_->get(prefixDir_+"/"+globalFolder_ +"/DeadChannelPercentage_EndcapNegative");  
	    regionME->setBinContent((*meItr).first, (-1*(*itr).first ),(*meItr).second.first/(*meItr).second.second );
	  }
      }    
    }
  }
}

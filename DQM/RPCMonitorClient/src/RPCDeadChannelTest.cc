/*
 *  \author Anna Cimmino
 */
#include "DQM/RPCMonitorDigi/interface/utils.h"
#include <DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h>
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DataFormats
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

// Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <map>
#include <sstream>
//#include <math.h>

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
 
 stringstream histoName, histoTitle;

 for (int i = -4; i<=4;i++ ){
   if (i>-3 && i<3){//wheels
     histoName.str("");
     histoName<<"DeadChannels_Wheel"<<i;
     histoTitle.str("");
     histoTitle<<"DeadChannels for Wheel "<<i;
     if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
     }
  
     me = dbe_->book2D(histoName.str().c_str(), histoTitle.str().c_str(), 12, 0.5, 12.5, 21, 0.5, 21.5);
     for(int bin =1; bin<13;bin++) {
       histoName.str("");
       histoName<<"Sec"<<bin;
       me->setBinLabel(bin,histoName.str().c_str(),1);
     }
     histoName.str("");
     histoName<<"ClusterSize_vs_AliveStrips_Wheel"<<i;
     histoTitle.str("");
     histoTitle<<"ClusterSize vs AliveStrips Wheel "<<i;
     if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
     }
     
     me = dbe_->book2D(histoName.str().c_str(), histoTitle.str().c_str(), 12, 0.5, 12.5, 21, 0.5, 21.5);


     for(int bin =1; bin<13;bin++) {
       histoName.str("");
       histoName<<"Sec"<<bin;
       me->setBinLabel(bin,histoName.str().c_str(),1);
     }
   }//end wheels


   histoName.str("");
   histoName<<"DeadChannels_Disk"<<i;
   histoTitle.str("");
   histoTitle<<"DeadChannels for Disk "<<i;
   if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }
   me = dbe_->book2D(histoName.str().c_str(), histoTitle.str().c_str(), 6, 0.5, 6.5, 54, 0.5, 54.5);
   
   for(int bin =1; bin<7;bin++) {
     histoName.str("");
     histoName<<"Sec"<<bin;
     me->setBinLabel(bin,histoName.str().c_str(),1);
   }

   histoName.str("");
   histoName<<"ClusterSize_vs_AliveStrips_Disk"<<i;
   histoTitle.str("");
   histoTitle<<"ClusterSize vs AliveStrips Disk "<<i;
   if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }

   me = dbe_->book2D(histoName.str().c_str(), histoTitle.str().c_str(), 6, 0.5, 6.5, 54, 0.5, 54.5);
 
   
   for(int bin =1; bin<7;bin++) {
     histoName.str("");
     histoName<<"Sec"<<bin;
     me->setBinLabel(bin,histoName.str().c_str(),1);
   }
 }
 
 histoName.str("");
 histoName<<"DeadChannelPercentageBarrel";
 if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
       dbe_->removeElement(me->getName());
 }
 me = dbe_->book2D(histoName.str().c_str(), "Dead Channel Fraction in Barrel", 12, 0.5, 12.5, 5, -2.5, 2.5);

 for(int xbin =1; xbin<13; xbin++) {
   histoName.str("");
   histoName<<"Sec"<<xbin;
   me->setBinLabel(xbin,histoName.str().c_str(),1);
 }
 for(int ybin =1; ybin<5; ybin++) {
   histoName.str("");
   histoName<<"Wheel"<<(ybin-3);
   me->setBinLabel(ybin,histoName.str().c_str(),2);
 }
 
 histoName.str("");
 histoName<<"DeadChannelPercentageEndcapPositive";
 if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
   dbe_->removeElement(me->getName());
 }
 me = dbe_->book2D(histoName.str().c_str(), "Dead Channel Fraction in Endcap+", 6, 0.5, 6.5, 4, -2, 2);

 for(int xbin =1; xbin<7; xbin++) {
   histoName.str("");
   histoName<<"Sec"<< xbin;
   me->setBinLabel( xbin,histoName.str().c_str(),1);
 }
 for(int ybin =1; ybin<5; ybin++) {
   histoName.str("");
   histoName<<"Disk"<<ybin;
   me->setBinLabel(ybin,histoName.str().c_str(),2);
 }
 
 histoName.str("");
 histoName<<"DeadChannelPercentageEndcapNegative";
 if ( me = dbe_->get(prefixDir_+"/"+globalFolder_ +"/"+ histoName.str()) ) {
   dbe_->removeElement(me->getName());
 }
 me = dbe_->book2D(histoName.str().c_str(), "Dead Channel Fraction in Endcap-", 6, 0.5, 6.5,4, -2, 2); 

 for(int xbin =1; xbin<7; xbin++) {
   histoName.str("");
   histoName<<"Sec"<< xbin;
   me->setBinLabel( xbin,histoName.str().c_str(),1);
 }
 for(int ybin =1; ybin<5; ybin++) {
   histoName.str("");
   histoName<<"Disk-"<<ybin;
   me->setBinLabel(ybin,histoName.str().c_str(),2);
 } 

}

void RPCDeadChannelTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {}

//called at each event
void RPCDeadChannelTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c){}


void RPCDeadChannelTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {
 
  edm::LogVerbatim ("deadChannel") <<"[RPCDeadChannelTest]: End of LS transition, performing the DQM client operation";
  
  // counts number of lumiSegs 
  int nLumiSegs = lumiSeg.id().luminosityBlock();
    
  //check some statements and prescale Factor
  if(nLumiSegs%prescaleFactor_ == 0) {
 
    ESHandle<RPCGeometry> rpcGeo;
    iSetup.get<MuonGeometryRecord>().get(rpcGeo);
 
    map<int, map< int ,  pair<float,float> > >  barrelMap, endcapMap;
    stringstream meName;
    //Loop on chambers
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
	
	 MonitorElement * myGlobalMe;
	MonitorElement * myGlobalMe2;

	 const QReport * theOccupancyQReport = myMe->getQReport("DeadChannel_0");  
	 if(!theOccupancyQReport) continue;

	 vector<dqm::me_util::Channel> badChannels = theOccupancyQReport->getBadChannels();
	
	 if (detId.region()==0) {
	   barrelMap[detId.ring()][detId.sector()].first += badChannels.size();
	   barrelMap[detId.ring()][detId.sector()].second += (*r)->nstrips() ;
	   meName.str("");
	   meName<<prefixDir_+"/"+ globalFolder_+"/DeadChannels_Wheel"<<detId.ring();
	 }else{
	   endcapMap[detId.region()*detId.station()][detId.sector()].first +=  badChannels.size();
	   endcapMap[detId.region()*detId.station()][detId.sector()].second+=(*r)->nstrips() ;
	   meName.str("");
	   meName<<prefixDir_+"/"+ globalFolder_+"/DeadChannels_Disk"<<detId.region()*detId.station();
	 }
	 myGlobalMe = dbe_->get(meName.str());
	 if (!myGlobalMe)continue;
	 rpcdqm::utils prova;
	 int nr = prova.detId2RollNr(detId);
	 myGlobalMe->setBinContent(detId.sector(),nr, badChannels.size()*100/(*r)->nstrips() );

	 string Yaxis=RPCname.name();
	 if (detId.region()==0){
	   Yaxis.erase (1,1);
	   Yaxis.erase(0,3);
	   Yaxis.replace(Yaxis.find("S"),4,"");
	   Yaxis.erase(Yaxis.find("_")+2,8);
	 }else{
	   Yaxis.erase(0,8);
	 }

	 myGlobalMe->setBinLabel(nr, Yaxis, 2);
	 if (detId.region()==0){
	   meName.str("");
	   meName<<prefixDir_+"/"+ globalFolder_+"/ClusterSize_vs_AliveStrips_Wheel"<<detId.ring();
	   myGlobalMe = dbe_->get(meName.str());
	   meName.str("");
	   meName<<prefixDir_+"/"+ globalFolder_+"/ClusterSize_meanValue_Wheel_"<<detId.ring();
	   myGlobalMe2 = dbe_->get(meName.str());

	   if(badChannels.size()!=(*r)->nstrips() )
	     myGlobalMe->setBinContent(detId.sector(),nr, (myGlobalMe2->getBinContent(detId.sector(),nr))/((*r)->nstrips()-badChannels.size()) );
	   else 
	     myGlobalMe->setBinContent(detId.sector(),nr, 100 ); 
	 }

	 myGlobalMe->setBinLabel(nr, Yaxis, 2);
       }//End loop on rolls in given chambers
    }
  }//End loop on chamber

  this->fillDeadChannelHisto(barrelMap, 0);
  
  this->fillDeadChannelHisto(endcapMap, 1);
  }
}
 
void RPCDeadChannelTest::endRun(const Run& r, const EventSetup& c){}

void RPCDeadChannelTest::endJob(){}

//Fill report summary
void  RPCDeadChannelTest::fillDeadChannelHisto(const map<int,map<int,pair<float,float> > > & sumMap, int region){

  MonitorElement *   regionME=NULL;

  map<int,map<int,pair<float,float> > >::const_iterator itr;
 
  if (sumMap.size()!=0){
    for (itr=sumMap.begin(); itr!=sumMap.end(); itr++){
      for (map< int ,  pair<float,float> >::const_iterator meItr = (*itr).second.begin(); meItr!=(*itr).second.end();meItr++){
	  if (region==0){
	    regionME = dbe_->get(prefixDir_+"/"+globalFolder_ +"/DeadChannelPercentageBarrel");
	    regionME->setBinContent((*meItr).first, (*itr).first + 3,(*meItr).second.first/(*meItr).second.second );
	  }else {
	    regionME = dbe_->get(prefixDir_+"/"+globalFolder_ +"/DeadChannelPercentageEndcapPositive");  
	    regionME->setBinContent((*meItr).first, (*itr).first ,(*meItr).second.first/(*meItr).second.second );

	    regionME = dbe_->get(prefixDir_+"/"+globalFolder_ +"/DeadChannelPercentageEndcapNegative");  
	    regionME->setBinContent((*meItr).first, (-1*(*itr).first ),(*meItr).second.first/(*meItr).second.second );
	  }
      }    
    }
  }
}

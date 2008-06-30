/*  \author Anna Cimmino*/
#include <string>
#include <sstream>
#include <map>
#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DQM Services
#include "DQMServices/Core/interface/DQMStore.h"
//DataFormats
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
//Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace edm;
using namespace std;
RPCEventSummary::RPCEventSummary(const ParameterSet& ps ){
  LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Constructor";

  enableReportSummary_ = ps.getUntrackedParameter<bool>("EnableSummaryReport",true);
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 10);
  eventInfoPath_ = ps.getUntrackedParameter<string>("EventInfoPath", "RPC/EventInfo");
  prefixDir_ = ps.getUntrackedParameter<string>("RPCPrefixDir", "RPC/RecHits");
  verbose_=ps.getUntrackedParameter<unsigned int>("VerboseLevel", 0);

}

RPCEventSummary::~RPCEventSummary(){
  LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Destructor ";

  dbe_=0;
}

void RPCEventSummary::beginJob(const EventSetup& iSetup){
 LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin job ";
  
 dbe_ = Service<DQMStore>().operator->();

 dbe_->setVerbose(verbose_);
}

void RPCEventSummary::beginRun(const Run& r, const EventSetup& c){
 LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin run";

 MonitorElement* me;
 dbe_->setCurrentFolder(eventInfoPath_);

 //a global summary float [0,1] providing a global summary of the status 
 //and showing the goodness of the data taken by the the sub-system 
 string histoName="reportSummary";
 if ( me = dbe_->get(eventInfoPath_ +"/"+ histoName) ) {
    dbe_->removeElement(me->getName());
  }

  me = dbe_->bookFloat(histoName);
  me->Fill(-1);

  //TH2F ME providing a map of values[0-1] to show if problems are localized or distributed
  if ( me = dbe_->get(eventInfoPath_ + "/reportSummaryMap") ) {
     dbe_->removeElement(me->getName());
  }
  me = dbe_->book2D("reportSummaryMap", "RPC Report Summary Map", 15, -7.5, 7.5, 12, 0.5 ,12.5);
  
  //customize the 2d histo
  stringstream BinLabel;
  for (int i= 1 ; i<=15; i++){
    BinLabel.str("");
    if(i<13){
      BinLabel<<"Sec"<<i;
       me->setBinLabel(i,BinLabel.str(),2);
    } 

    BinLabel.str("");
    if(i<5)
      BinLabel<<"Disk"<<i-5;
    else if(i>11)
      BinLabel<<"Disk"<<i-11;
    else if(i==11 || i==5)
      BinLabel.str("");
    else
      BinLabel<<"Wheel"<<i-8;
 
     me->setBinLabel(i,BinLabel.str(),1);
  }

  //fill the histo with "1" --- just for the moment
  for(int i=1; i<=15; i++){
     for (int j=1; j<=12; j++ ){
       me->setBinContent(i,j,0);
     }
   }

 //the reportSummaryContents folder containins a collection of ME floats [0-1] (order of 5-10)
 // which describe the behavior of the respective subsystem sub-components.
 dbe_->setCurrentFolder(eventInfoPath_+ "/reportSummaryContents");

  segmentNames.push_back("RPC_Wheel-2");
  segmentNames.push_back("RPC_Wheel-1");
  segmentNames.push_back("RPC_Wheel0");
  segmentNames.push_back("RPC_Wheel1");
  segmentNames.push_back("RPC_Wheel2");

  segmentNames.push_back("RPC_Disk-4");
  segmentNames.push_back("RPC_Disk-3");
  segmentNames.push_back("RPC_Disk-2");
  segmentNames.push_back("RPC_Disk-1");
  segmentNames.push_back("RPC_Disk1");
  segmentNames.push_back("RPC_Disk2");
  segmentNames.push_back("RPC_Disk3");
  segmentNames.push_back("RPC_Disk4");
  
  //  segmentNames.push_back("RPC_DataIntegrity");
  // segmentNames.push_back("RPC_Timing");

  for(int i=0; i<segmentNames.size(); i++){
    if ( me = dbe_->get(eventInfoPath_ + "/reportSummaryContents/" +segmentNames[i]) ) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->bookFloat(segmentNames[i]);
    me->Fill(-1);
  }
}

void RPCEventSummary::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCEventSummary::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCEventSummary::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCEventSummary]: End of LS transition, performing DQM client operation";
  
  // counts number of lumiSegs 
  nLumiSegs_ = lumiSeg.id().luminosityBlock();
  //check some statements and prescale Factor
  if(!enableReportSummary_  ||  nLumiSegs_%prescaleFactor_ != 0) return;

  ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
 
  map<int, map< int ,  pair<float,float> > >  barrelMap, endcapPlusMap, endcapMinusMap;
    
  //Loop on chambers
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
       RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
       std::vector< const RPCRoll*> roles = (ch->rolls());
       int ty=1;
       //Loop on rolls in given chamber
       for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	 RPCDetId detId = (*r)->id();
	 //Get Occupancy ME for roll
	 RPCGeomServ RPCname(detId);
	 RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
	 MonitorElement * myMe = dbe_->get(prefixDir_+"/"+ folderStr->folderStructure(detId)+"/Occupancy_"+RPCname.name()); 
	 if (!myMe)continue;
	 const QReport * theOccupancyQReport = myMe->getQReport("DeadChannel_0");  
	 if(!theOccupancyQReport) continue;
	 vector<dqm::me_util::Channel> badChannels = theOccupancyQReport->getBadChannels();
	 float goodFraction =((*r)->nstrips() - badChannels.size())/(*r)->nstrips();		  
	 if (detId.region()==0) {
	   barrelMap[detId.ring()][detId.sector()].first += goodFraction;
	   barrelMap[detId.ring()][detId.sector()].second++ ;
	 }else if(detId.region()==-1){
	   endcapMinusMap[detId.station()][detId.sector()].first +=  badChannels.size();
	   endcapMinusMap[detId.station()][detId.sector()].second+=(*r)->nstrips() ;
	 }else {
	   endcapPlusMap[detId.station()][detId.sector()].first +=  badChannels.size();
	   endcapPlusMap[detId.station()][detId.sector()].second+=(*r)->nstrips();
	 }
	 ty++;      
       }//End loop on rolls in given chambers
    }
  }//End loop on chamber
  
  //fill report Summary MEs
  MonitorElement *   reportSummary = dbe_->get(eventInfoPath_ +"/reportSummary");
  MonitorElement *   reportSummaryMap = dbe_->get(eventInfoPath_ +"/reportSummaryMap");
  // MonitorElement *   reportSummaryBarrel = dbe_->get(eventInfoPath_ +"/reportSummaryContents/RPCSummaryBarrel");

  
  //Loop on barrel report summary data 
  map<int,map<int,pair<float,float> > >::const_iterator itr;
  stringstream meName;

  float allRolls=0;
  float allGood=0;

  for (itr=barrelMap.begin(); itr!=barrelMap.end(); itr++){
    float wheelRolls=0; 
    float wheelGood=0;
    for (map< int ,  pair<float,float> >::const_iterator meItr = (*itr).second.begin(); meItr!=(*itr).second.end();meItr++){
      //Fill report summary map, a TH2F ME.
      if ((*meItr).second.second != 0) 
	reportSummaryMap->setBinContent((*itr).first+8,(*meItr).first, ((*meItr).second.first/(*meItr).second.second) ); 
      else reportSummaryMap->setBinContent((*itr).first+8,(*meItr).first,-1);
      wheelGood += (*meItr).second.first;
      wheelRolls += (*meItr).second.second;
    }
    allGood += wheelGood;
    allRolls +=  wheelRolls ;
    //Fill wheel/disk report summary
    meName.str("");
    meName<<eventInfoPath_<<"/reportSummaryContents/RPC_Wheel"<<(*itr).first;
    MonitorElement *   reportSummaryContents = dbe_->get( meName.str());
    if (wheelRolls != 0)  reportSummaryContents->Fill(wheelGood/wheelRolls);
    else reportSummary->Fill(-1);
  }
  //Fill report summary
  if (allRolls!=0)   reportSummary->Fill( allGood/allRolls);
  else reportSummary->Fill(-1);
}




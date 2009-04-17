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
  verbose_=ps.getUntrackedParameter<bool>("VerboseLevel", 0);
  minHitsInRoll_=ps.getUntrackedParameter<unsigned int>("MinimunHitsPerRoll", 2000);
 tier0_=ps.getUntrackedParameter<bool>("Tier0", false);

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

 nLumiSegs_=0;

 MonitorElement* me;
 dbe_->setCurrentFolder(eventInfoPath_);

 //a global summary float [0,1] providing a global summary of the status 
 //and showing the goodness of the data taken by the the sub-system 
 string histoName="reportSummary";
 if ( me = dbe_->get(eventInfoPath_ +"/"+ histoName) ) {
    dbe_->removeElement(me->getName());
  }

  me = dbe_->bookFloat(histoName);
  me->Fill(1);

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
       if(i==5 || i==11 || (j>6 && (i<6 || i>10)))    
	 me->setBinContent(i,j,-1);//bins that not correspond to subdetector parts
       else     
	 me->setBinContent(i,j,1);
     }
   }

 //the reportSummaryContents folder containins a collection of ME floats [0-1] (order of 5-10)
 // which describe the behavior of the respective subsystem sub-components.
  dbe_->setCurrentFolder(eventInfoPath_+ "/reportSummaryContents");
  
  stringstream segName;
  for(int i=-4; i<=4; i++){
    if(i>-3 && i<3) {
      segName.str("");
      segName<<"RPC_Wheel"<<i;
      segmentNames.push_back(segName.str());
    }
    if(i==0) continue;
    segName.str("");
    segName<<"RPC_Disk"<<i;
    segmentNames.push_back(segName.str());
  }
  
  //  segmentNames.push_back("RPC_DataIntegrity");
  // segmentNames.push_back("RPC_Timing");

  for(unsigned int i=0; i<segmentNames.size(); i++){
    if ( me = dbe_->get(eventInfoPath_ + "/reportSummaryContents/" +segmentNames[i]) ) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->bookFloat(segmentNames[i]);
    me->Fill(1);
  }
}

void RPCEventSummary::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCEventSummary::analyze(const Event& iEvent, const EventSetup& c) {
  nLumiSegs_=iEvent.luminosityBlock();
}

void RPCEventSummary::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCEventSummary]: End of LS transition, performing DQM client operation";

  if (tier0_) return;

  // counts number of lumiSegs 
   nLumiSegs_ = lumiSeg.id().luminosityBlock();
   //nLumiSegs_++;

  //check some statements and prescale Factor
  if(enableReportSummary_  &&  (nLumiSegs_%prescaleFactor_ == 0)) {
 
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
	 
	 //check for enough statistics
	 if (myMe->getEntries() < minHitsInRoll_) continue;

	 const QReport * theOccupancyQReport = myMe->getQReport("DeadChannel_0");  
	 if(!theOccupancyQReport) continue;
	 vector<dqm::me_util::Channel> badChannels = theOccupancyQReport->getBadChannels();
	 float goodFraction =((*r)->nstrips() - badChannels.size())/(*r)->nstrips();		  
	 if (detId.region()==0) {
	   barrelMap[detId.ring()][detId.sector()].first += goodFraction;
	   barrelMap[detId.ring()][detId.sector()].second++ ;
	 }else if(detId.region()==-1){
	   endcapMinusMap[-1 * detId.station()][detId.sector()].first +=  goodFraction;
	   endcapMinusMap[-1 * detId.station()][detId.sector()].second++ ;
	 }else {
	   endcapPlusMap[detId.station()][detId.sector()].first += goodFraction;
	   endcapPlusMap[detId.station()][detId.sector()].second++;
	 }
	 ty++;      
       }//End loop on rolls in given chambers
    }
  }//End loop on chamber
  
  //clear counters
  allRolls_=0;
  allGood_=0;
  
  this->fillReportSummary(barrelMap, 0);
  
  this->fillReportSummary(endcapPlusMap, 1);
  
  this->fillReportSummary(endcapMinusMap, -1);

  //Fill report summary
  MonitorElement *   reportSummary = dbe_->get(eventInfoPath_ +"/reportSummary");
  if(reportSummary == NULL) return;

  if (allRolls_!=0)   reportSummary->Fill(allGood_/allRolls_);
  else reportSummary->Fill(-1);
  }
}


//Fill report summary
void  RPCEventSummary::fillReportSummary(const map<int,map<int,pair<float,float> > > & sumMap, int region){

  MonitorElement *   reportSummaryMap = dbe_->get(eventInfoPath_ +"/reportSummaryMap");
  
  string path;
  int binOffSet=0;

  if (region==0){
    path="/reportSummaryContents/RPC_Wheel";
    binOffSet=8;
  }else if (region==1){
    path="/reportSummaryContents/RPC_Disk";
    binOffSet=11;
  }else if (region==-1){
    path="/reportSummaryContents/RPC_Disk-";
    binOffSet=5;
  }

  map<int,map<int,pair<float,float> > >::const_iterator itr;
  stringstream meName;

  if (sumMap.size()!=0){
    //Loop on  report summary data 
    for (itr=sumMap.begin(); itr!=sumMap.end(); itr++){
      float Rolls=0; 
      float Good=0;
      for (map< int ,  pair<float,float> >::const_iterator meItr = (*itr).second.begin(); meItr!=(*itr).second.end();meItr++){
	//Fill report summary map, a TH2F ME.
	if ((*meItr).second.second != 0) {
	  //	  cout<<"i'm here"<<(*itr).first<<"_"<< region<<endl;
	  reportSummaryMap->setBinContent((*itr).first+binOffSet,(*meItr).first, ((*meItr).second.first/(*meItr).second.second) ); 
      }  else reportSummaryMap->setBinContent((*itr).first+binOffSet,(*meItr).first,-1);
	Good += (*meItr).second.first;
	Rolls += (*meItr).second.second;
      }
      allGood_ += Good;
      allRolls_ +=  Rolls ;
      //Fill wheel/disk report summary
      meName.str("");
      meName<<eventInfoPath_<<path<<(*itr).first;

      MonitorElement *   reportSummaryContents = dbe_->get( meName.str());
      if (Rolls != 0)  reportSummaryContents->Fill(Good/Rolls);
      else reportSummaryContents->Fill(-1);
    }  //End Loop on report summary data 
  }else{
    for (int j=0; j<=4; j++){
      meName.str("");
      meName<<eventInfoPath_<<path<<j;

      MonitorElement *   reportSummaryContents = dbe_->get( meName.str());
      if ( reportSummaryContents == NULL) continue;
	reportSummaryContents->Fill(-1);
      for(int h =1; h<=12; h++){
	if(region!=0 && h>6) break; //endcap has only 6 sectors
	if (region == -1) reportSummaryMap->setBinContent(-j+binOffSet,h, -1); 
	else reportSummaryMap->setBinContent(j+binOffSet,h, -1); 
      }
    }   
  }
}

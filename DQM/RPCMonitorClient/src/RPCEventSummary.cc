/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/04/25 14:24:57 $
 *  $Revision: 1.1 $
 *  \author Anna Cimmino
 */

#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Event.h>

//DQM Services
#include "DQMServices/Core/interface/DQMStore.h"

//DataFormats
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

// Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

RPCEventSummary::RPCEventSummary(const ParameterSet& ps ){
 
  LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Constructor";

  //get event info folder (remenber: this module must run after EventInfo!!!!
  eventInfoPath_ = ps.getUntrackedParameter<string>("EventInfoPath", "RPC/EventInfo");

  prefixDir_ = ps.getUntrackedParameter<string>("RPCPrefixDir", "RPC/RecHits");


  enableDetectorSegmentation_ = ps.getUntrackedParameter<bool>("EnableDetectorSegmentation", true);

}


RPCEventSummary::~RPCEventSummary(){
  
  LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Destructor ";
  
  dbe_=0;
}

//called only once
void RPCEventSummary::beginJob(const EventSetup& iSetup){

 LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin job ---------------------------------------------";
  
 // get hold of back-end interface
 dbe_ = Service<DQMStore>().operator->();

}


//Begin Run
void RPCEventSummary::beginRun(const Run& r, const EventSetup& c){

 LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Begin run";

//set to -1 the event summary info (-1 = no info yet ... but it's coming....)

 MonitorElement* me;
 dbe_->setCurrentFolder(eventInfoPath_);

 string histoName="reportSummary";

 //a global summary float [0,1] providing a global summary of the status 
 //and showing the goodness of the data taken by the the sub-system 
 if ( me = dbe_->get(eventInfoPath_ +"/"+ histoName) ) {
    dbe_->removeElement(me->getName());
  }

  me = dbe_->bookFloat(histoName);
  me->Fill(1);

  if ( me = dbe_->get(eventInfoPath_ + "/reportSummaryMap") ) {
     dbe_->removeElement(me->getName());
  }
  me = dbe_->book2D("reportSummaryMap", "reportSummaryMap", 100, 0, 100, 100, 0, 100);
  me->setAxisTitle("jphi", 1);
  me->setAxisTitle("jeta", 2);

  //fill the histo with "-1" --- just for the moment
  for(int i=0; i<100; i++){
    for (int j=0; j<100; j++ ){
      me->setBinContent(i,j,1);
    }
  }


 dbe_->setCurrentFolder(eventInfoPath_+ "/reportSummaryContents");

 //the reportSummaryContents folder containins a collection of ME floats [0-1] (order of 5-10)
 // which describe the behavior of the respective subsystem sub-component.
  segmentNames.push_back("RPCEndcap-");
  segmentNames.push_back("RPCBarrel");
  segmentNames.push_back("RPCEndcap+");

  for(int i=0; i<segmentNames.size(); i++){
    if ( me = dbe_->get(eventInfoPath_ + "/reportSummaryContents/" +segmentNames[i]) ) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->bookFloat(segmentNames[i]);
    me->Fill(1);
  }
}
  

void RPCEventSummary::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  LogVerbatim ("rpceventsummary") <<"[RPCEventSummary]: Begin of LS transition";

  // Get the run number & luminosity block
  // run = lumiSeg.run();
  //  lumiBlock=lumiSeg.luminosityBlock();
}


//called at each event
void RPCEventSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& c){
  
  LogVerbatim ("rpceventsummary") << "[RPCEventSummary]:  analyzer called ";

  /// get didgi collection for event
  Handle<RPCDigiCollection> rpcdigis;
  iEvent.getByType(rpcdigis);

  //get new Histos
  //loop on digi collection 
  for( RPCDigiCollection::DigiRangeIterator collectionItr=rpcdigis->begin();collectionItr!=rpcdigis->end(); ++collectionItr){
    RPCDetId detId=(*collectionItr ).first; 
    map<RPCDetId ,string>::iterator meItr = meCollection.find(detId);
    if (meItr == meCollection.end() || (meCollection.size()==0)) {
      
      meCollection[detId] = getMEName(detId);
    }
  }// end loop on digi collection
}      


void RPCEventSummary::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {  
  edm::LogVerbatim ("rpceventsummary") <<"[RPCEventSummary]: End of LS transition, performing the DQM client operation";
  
  // counts number of lumiSegs 
  //  nLumiSegs_ = lumiSeg.id().luminosityBlock();
  vector <float> badChannelsInSegment; 
  vector <float> channelsInSegment;

  for(int i =0 ;i<3; i++){
    badChannelsInSegment.push_back(0);
    channelsInSegment.push_back(0);
  }

  float totalBadChannels=0;
  float  totalChannels=0;

 /* 
  //check some statements and prescale Factor
  if( !getQualityTestsFromFile ||  nLumiSegs_%prescaleFactor != 0 ) return;
 */

  // Occupancy test - the test was already performed and attached to the ME. Here we only retreive the results
  string OccupancyTestName ="DeadChannel_0"; 
 
  //Loop over Histos
  for(std::map<RPCDetId, string>::const_iterator hOccIt = meCollection.begin();hOccIt != meCollection.end(); hOccIt++) {
 
    //get hold of ME
    MonitorElement * myMe = dbe_->get((*hOccIt).second);//(*hOccIt).second);

    //get quality report
    if (!myMe) continue; 
    const QReport * theOccupancyQReport = myMe->getQReport(OccupancyTestName);
  
    if(theOccupancyQReport) {
      vector<dqm::me_util::Channel> badChannels = theOccupancyQReport->getBadChannels();
      
      totalBadChannels+=badChannels.size();
      totalChannels += myMe->getNbinsX();
     
      //check if region is barrel or endcap
      if (enableDetectorSegmentation_){
	  badChannelsInSegment[(*hOccIt).first.region()+1]+=badChannels.size();
	  channelsInSegment[(*hOccIt).first.region()+1]+= myMe->getNbinsX();
      }
    }
    else 
      LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: No Event Summary from  "<<(*hOccIt).second;       
  }//end loop over Histos
  

  //calulate total alive fraction of strips to give an event summary status label (number from 0 -> 1)
  float fraction = -1;
  if(totalChannels!=0) fraction = ((int)(((totalChannels-totalBadChannels)/totalChannels)*100))/100.0;
  LogVerbatim ("rpceventsummary") << "[RPCEventSummary]: Total Alive Detector Fraction:"<<fraction;

  MonitorElement * meEventSummary = dbe_->get(eventInfoPath_+"/reportSummary");
  if(meEventSummary) meEventSummary->Fill(fraction);
  
  if (enableDetectorSegmentation_){
    stringstream meSegmentName;
    
    //loop on detector segments
        for (int i = 0 ; i<channelsInSegment.size(); i++){
      float fraction1=-1;
      if (channelsInSegment[i]!=0) fraction1 = ((int)(((channelsInSegment[i]-badChannelsInSegment[i])/channelsInSegment[i])*100))/100.0;

      meSegmentName.str("");
      meSegmentName<<eventInfoPath_<<"/reportSummaryContents/"<<segmentNames[i];
      MonitorElement * meSegment = dbe_->get(meSegmentName.str());
      if(meSegment){ meSegment->Fill(fraction1);
      }
    }//end loop on detector segments
  }
}

// find ME name using RPC geometry
string RPCEventSummary::getMEName(RPCDetId & detId) {    
  string regionName;
  string ringType;
  
  if(detId.region() ==  0) {
    regionName="Barrel";
    ringType="Wheel";
  }else{
    ringType="Disk";
    if(detId.region() == -1) regionName="Encap-";
    if(detId.region() ==  1) regionName="Encap+";
  }
  
  //get hold of RPCName  
  RPCGeomServ RPCname(detId);
  
  /// Get histos
  stringstream myStream ;
  myStream <<prefixDir_<<"/" <<regionName<<"/"<<ringType<<"_"<<detId.ring()<<"/station_"<<detId.station()<<"/sector_"<<detId.sector()<<"/Occupancy_"<<RPCname.name();
  
  return myStream.str();
}

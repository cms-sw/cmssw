/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/07 18:14:23 $
 *  $Revision: 1.1 $
 *  \author
 */


#include <DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h>

// Framework
/*#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>*/
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOldReceiver.h"

//DataFormats

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
//#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

// Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
//#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//#include "Geometry/Records/interface/MuonGeometryRecord.h"


#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
#include <fstream>
#include <time.h>

using namespace edm;
using namespace std;

RPCDeadChannelTest::RPCDeadChannelTest(const edm::ParameterSet& ps ){
 
  edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: Constructor";

  parameters = ps;
  getQualityTestsFromFile = parameters.getUntrackedParameter<bool> ("getQualityTestsFromFile",false);
  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);
  referenceOldDeadChannels = parameters.getUntrackedParameter<bool> ("getReferenceFile",false);
}


RPCDeadChannelTest::~RPCDeadChannelTest(){

  edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: analyzed " << nevents << " events";
  
  delete mui_;
  delete qtHandler;
}

//called only once
void RPCDeadChannelTest::beginJob(const edm::EventSetup& c){

 edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: Begin job";
  
 // get hold of back-end interface
 dbe_ = Service<DQMStore>().operator->();
  
  mui_ = new DQMOldReceiver();

  dbe_->setVerbose(1);
  
  qtHandler = new QTestHandle();
  
  //configure quality test using xml file
  if (getQualityTestsFromFile)
    qtHandler->configureTests(parameters.getUntrackedParameter<string> ("qtList","QualityTests.xml"),dbe_);

  //get local date and time
  time_t t = time(0);
  strftime( dateTime, sizeof(dateTime), "%Y_%m_%d_%H_%M_%S.txt", localtime(&t));

  //open txt file
  myfile.open(dateTime, ios::app);
  myfile<<"RUN LUMIBLOCK CHAMBER STRIP TAG RefRUN\n";
  
  //get reference file
  if (referenceOldDeadChannels)
    referenceFile_.open(parameters.getUntrackedParameter<string> ("referenceFileName","reference.txt").c_str());
}



void RPCDeadChannelTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("deadChannel") <<"[RPCDeadChannelTest]: Begin of LS transition";

  // Get the run number & luminosity block
  run = lumiSeg.run();
  lumiBlock=lumiSeg.luminosityBlock();
}


//called at each event
void RPCDeadChannelTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c){
  nevents++;
  edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: "<<nevents<<" events";

  /// get didgi collection for event
  edm::Handle<RPCDigiCollection> rpcdigis;
  iEvent.getByType(rpcdigis);

  //get new Histos
  //loop on digi collection 
  for( RPCDigiCollection::DigiRangeIterator collectionItr=rpcdigis->begin(); 
       collectionItr!=rpcdigis->end(); ++collectionItr){
    RPCDetId detId=(*collectionItr ).first; 
    std::map<RPCDetId ,MonitorElement*>::iterator meItr = meCollection.find(detId);
    if (meItr == meCollection.end() || (meCollection.size()==0)) {
      
      meCollection[detId] = getMEs(detId);
    }
  }// end loop on digi collection


}      



void RPCDeadChannelTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
 
  
  edm::LogVerbatim ("deadChannel") <<"[RPCDeadChannelTest]: End of LS transition, performing the DQM client operation";
  
  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
  
  //check some statements and prescale Factor
  if( !getQualityTestsFromFile ||  nLumiSegs%prescaleFactor != 0 ) return;
 
  edm::LogVerbatim ("deadChannel") <<"[RPCDeadChannelTest]: "<<nLumiSegs<<" updates";

  
  edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: Occupancy tests results";

  //always needed
  mui_->doMonitoring();

  //attach qTest. Done here because new ME can appear while processing data
  edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: Attaching QTests";
  qtHandler->attachTests(dbe_);

  //run qtest. All tests in xml file will run.   
  edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: Running QTests";
  dbe_->runQTests();

  // Occupancy test
  string OccupancyTestName = parameters.getUntrackedParameter<string>("OccTestName","DeadChannel_0"); 
  int deadchannel;
  string line,referenceRun,referenceLumiBlock,referenceRoll,referenceStrip, referenceTag, lastreferenceRun,nameRoll;
  //Loop over Histos
  for(std::map<RPCDetId, MonitorElement*>::const_iterator hOccIt = meCollection.begin();
      hOccIt != meCollection.end();
      hOccIt++) {
    if((*hOccIt).second->getEntries() > (prescaleFactor-1 )){
      const QReport * theOccupancyQReport = (*hOccIt).second->getQReport(OccupancyTestName);
      if(theOccupancyQReport) {
	vector<dqm::me_util::Channel> badChannels = theOccupancyQReport->getBadChannels();
	//loop on bad channels
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	     channel != badChannels.end(); channel++) {
	  //get roll name
	  RPCGeomServ RPCname((*hOccIt).first);
	  nameRoll = RPCname.name();
	  
	  edm::LogError ("deadChannel") << "Chamber : "<<nameRoll<<" Bad occupancy channel: "<<(*channel).getBin()<<" Contents : "<<(*channel).getContents();
	  if(myfile.is_open())
	    {

	      deadchannel = (* channel).getBin();
	      //   deadchannel = new String(deadcha);
	      if(referenceOldDeadChannels && referenceFile_.is_open()){

		int i=1; // line zero has titles -> start from line 1
		bool flag= false;
		//read reference file and find already known dead channels  
		referenceFile_.clear();// clear all status flags
		referenceFile_.seekg(0);//start reading file from the begining
		while (!referenceFile_.eof()&& !flag  )
		  {

		    referenceFile_>>referenceRun;
		    referenceFile_>>referenceLumiBlock;
		    referenceFile_>>referenceRoll;
		    referenceFile_>>referenceStrip;
		    referenceFile_>>referenceTag;
		    referenceFile_>>lastreferenceRun;
		    int p= atoi(referenceStrip.c_str());
		    if (referenceRoll == nameRoll && p==deadchannel)flag = true;
		    i++;
		  }
		if (flag){
		  myfile<<run<<" "<<lumiBlock<<" "<<nameRoll<<" "<<deadchannel<<" OLD-Reference Run:" + referenceRun +"\n";
		}
		else{ 
		  myfile<<run<<" "<<lumiBlock<<" "<<nameRoll<<" "<<deadchannel<<" NEW-Reference Run:" + referenceRun +"\n";
		}
	      }else {

		myfile<<run<<" "<<lumiBlock<<" "<<nameRoll<<" "<<deadchannel<<" No Info\n";
	      }
	      
	    }
	}//end loop on bad channels
	edm::LogWarning("deadChannel")<< "-------- Chamber : "<< nameRoll<<"  "<<theOccupancyQReport->getMessage()<<" ------- "<<theOccupancyQReport->getStatus(); 
      }
      else 
	edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: QReport for QTest "<<OccupancyTestName<<" is empty";
      
    }
  }//end loop over Histos
}


void RPCDeadChannelTest::endJob(){
  
  edm::LogVerbatim ("deadChannel") << "[RPCDeadChannelTest]: endjob called!";
  if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ) {
    stringstream runNumber; runNumber << run;
    string rootFile = "RPCDeadChannelTest_" + runNumber.str() + ".root";
    dbe_->save(parameters.getUntrackedParameter<string>("outputFile", rootFile));
  }

  //close txt file
  myfile.close();

  // dbe_->rmdir("RPC/Tests/RPCDeadChannel");  
}



MonitorElement* RPCDeadChannelTest::getMEs(RPCDetId & detId) {
  
  MonitorElement* me;
  
  std::string regionName;
  std::string ringType;
  if(detId.region() ==  0) {
    regionName="Barrel";
    ringType="Wheel";
  }else{
    ringType="Disk";
    if(detId.region() == -1) regionName="Encap-";
    if(detId.region() ==  1) regionName="Encap+";
  }
  
     
  RPCGeomServ RPCname(detId);
  char meId [328];
   
   /// Get histos
   sprintf(meId,"RPC/RecHits/%s/%s_%d/station_%d/sector_%d/Occupancy_%s",regionName.c_str(),ringType.c_str(),
	  detId.ring(),detId.station(),detId.sector(),RPCname.name().c_str());

     me = dbe_->get(meId);
   return me;
 }


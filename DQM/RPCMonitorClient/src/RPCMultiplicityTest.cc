/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/08 19:15:10 $
 *  $Revision: 1.2 $
 *  \author Anna Cimmino
 */


#include <DQM/RPCMonitorClient/interface/RPCMultiplicityTest.h>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM Services
#include "DQMServices/Core/interface/DQMStore.h"

//DataFormats
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

// Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
#include <fstream>
#include <time.h>

using namespace edm;
using namespace std;

RPCMultiplicityTest ::RPCMultiplicityTest (const ParameterSet& ps ){
 
  edm::LogVerbatim ("deadChannel") << "[RPCMultiplicityTest ]: Constructor";

  parameters = ps;
  //  getQualityTestsFromFile = parameters.getUntrackedParameter<bool> ("getQualityTestsFromFile",false);
  // prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);
  referenceOldChannels = parameters.getUntrackedParameter<bool> ("getReferenceFile_MultiplicityInRange",false);
}


RPCMultiplicityTest ::~RPCMultiplicityTest (){

  delete dbe_;
}

//called only once
void RPCMultiplicityTest ::beginJob(DQMStore * dbe){

   edm::LogVerbatim ("deadChannel") << "[RPCMultiplicityTest ]: Begin job";
  
 // get hold of back-end interface
 dbe_=dbe;

}


  //Begin Run
void RPCMultiplicityTest ::beginRun(const Run& r, const EventSetup& c){
  
//get local date and time
 time_t t = time(0);
 strftime( dateTime, sizeof(dateTime), "Digi_%Y_%m_%d_%H_%M_%S.txt", localtime(&t));
 
 //open txt output file
 myfile.open(dateTime, ios::app);
 myfile<<"RUN LUMIBLOCK CHAMBER #MEANDIGI TAG RefRUN\n";
 
 //get reference file
 if (referenceOldChannels)
   referenceFile_.open(parameters.getUntrackedParameter<string> ("referenceFileName_MultiplicityInRange","reference.txt").c_str()); 

}
  


void RPCMultiplicityTest ::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("numberofdigis") <<"[RPCMultiplicityTest ]: Begin of LS transition";

  // Get the run number & luminosity block
  run = lumiSeg.run();
  lumiBlock=lumiSeg.luminosityBlock();
}


//called at each event
void RPCMultiplicityTest ::analyze(const edm::Event& iEvent, const edm::EventSetup& c){
  // nevents++;
  // edm::LogVerbatim ("numberofdigis") << "[RPCMultiplicityTest ]: "<<nevents<<" events";

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



void RPCMultiplicityTest ::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
 
  
  edm::LogVerbatim ("numberofdigis") <<"[RPCMultiplicityTest ]: End of LS transition, performing the DQM client operation";
  
  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
 
 /* 
  //check some statements and prescale Factor
  if( !getQualityTestsFromFile ||  nLumiSegs%prescaleFactor != 0 ) return;
   */

  // Quality test - the test was already performed and attached to the ME. Here we only retreive the results
  string TestName = "MultiplicityInRange";
  int deadchannel;
  string line,referenceRun,referenceLumiBlock,referenceRoll,referenceStrip, referenceTag, lastreferenceRun,nameRoll;
  //Loop over Histos
  for(std::map<RPCDetId, MonitorElement*>::const_iterator hOccIt = meCollection.begin();
      hOccIt != meCollection.end();
      hOccIt++) {
    if((*hOccIt).second->getEntries() > (prescaleFactor-1 )){
      const QReport * theOccupancyQReport = (*hOccIt).second->getQReport(TestName );
      if(theOccupancyQReport) {
	//	vector<dqm::me_util::Channel> badChannels = theOccupancyQReport->getBadChannels();
	//loop on bad channels
	//	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	//   channel != badChannels.end(); channel++) {
	  //get roll name
	  RPCGeomServ RPCname((*hOccIt).first);
	  nameRoll = RPCname.name();
	  
	  //edm::LogError ("numberofdigis") << "Chamber : "<<nameRoll<<" Bad occupancy channel: "<<(*channel).getBin()<<" Contents : "<<(*channel).getContents();
	  if(myfile.is_open())
	    {

	      //  deadchannel = (* channel).getBin();

	      if(referenceOldChannels && referenceFile_.is_open()){
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
		    referenceFile_>>referenceTag;
		    referenceFile_>>lastreferenceRun;
		    int p= atoi(referenceStrip.c_str());
		    if (referenceRoll == nameRoll)flag = true;
		    i++;
		  }
		if (flag){
		  myfile<<run<<" "<<lumiBlock<<" "<<nameRoll<<" "<<(*hOccIt).second->getMean()<<" OLD-Reference Run:" + referenceRun +"\n";
		}
		else{ 
		  myfile<<run<<" "<<lumiBlock<<" "<<nameRoll<<" "<<(*hOccIt).second->getMean()<<" NEW-Reference Run:" + referenceRun +"\n";
		}
	      }else {

		myfile<<run<<" "<<lumiBlock<<" "<<nameRoll<<" "<<(*hOccIt).second->getMean()<<" No Info\n";
	      }
	      
	    }
	  //	}//end loop on bad channels
	edm::LogWarning("numberofdigis")<< "-------- Chamber : "<< nameRoll<<"  "<<theOccupancyQReport->getMessage()<<" ------- "<<theOccupancyQReport->getStatus(); 
      }
      else 
	edm::LogVerbatim ("numberofdigis") << "[RPCMultiplicityTest ]: QReport for QTest "<<TestName <<" is empty";
      
    }
  }//end loop over Histos
}



//End Run
void RPCMultiplicityTest ::endRun(const Run& r, const EventSetup& c){
  
 myfile.close();
}


void RPCMultiplicityTest ::endJob(){
  
  /*  edm::LogVerbatim ("numberofdigis") << "[RPCMultiplicityTest ]: endjob called!";
  if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ) {
    stringstream runNumber; runNumber << run;
    string rootFile = "RPCMultiplicityTest _" + runNumber.str() + ".root";
    dbe_->save(parameters.getUntrackedParameter<string>("outputFile", rootFile));
    }*/

  //close txt file
  //  myfile.close();

  // dbe_->rmdir("RPC/Tests/RPCDeadChannel");  
}


// find ME name using RPC geometry
MonitorElement* RPCMultiplicityTest::getMEs(RPCDetId & detId) {
  
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
   sprintf(meId,"RPC/RecHits/%s/%s_%d/station_%d/sector_%d/NumberOfDigi_%s",regionName.c_str(),ringType.c_str(),
	  detId.ring(),detId.station(),detId.sector(),RPCname.name().c_str());

     me = dbe_->get(meId);
   return me;
 }


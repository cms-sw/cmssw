// -*- C++ -*-
//
// Package:     SiStripHistoricInfoClient
// Class  :     SiStripHistoricInfoClient
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  dkcira
//         Created:  Thu Jun 15 09:32:49 CEST 2006
// $Id: SiStripHistoricInfoClient.cc,v 1.20 2008/03/02 00:07:43 dutta Exp $
//

#include "DQM/SiStripHistoricInfoClient/interface/SiStripHistoricInfoClient.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQMServices/Core/interface/DQMOldReceiver.h"
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

#include "xdata/Table.h"
#include "xdata/TableIterator.h"
#include "xdata/Integer.h"
#include "xdata/UnsignedInteger.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/UnsignedInteger64.h"
#include "xdata/UnsignedShort.h"
#include "xdata/UnsignedLong.h"
#include "xdata/Float.h"
#include "xdata/Double.h"
#include "xdata/Boolean.h"
#include "xdata/String.h"
#include "xdata/TimeVal.h"
#include "xdata/exdr/FixedSizeInputStreamBuffer.h"
#include "xdata/exdr/AutoSizeOutputStreamBuffer.h"
#include "xdata/exdr/Serializer.h"

#include "xdaq/ApplicationDescriptor.h"
#include "xdaq/ApplicationContext.h"

#include "TSQLServer.h"
#include "TSQLResult.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using namespace std;
using namespace cgicc;

//-----------------------------------------------------------------------------------------------
SiStripHistoricInfoClient::SiStripHistoricInfoClient(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "test",     // the name by which the collector identifies the client
		  "localhost",// the name of the computer hosting the collector
		  9090        // the port at which the collector listens
		  )
{
  // Instantiate a web interface:
  webInterface_p = new SiStripHistoricInfoWebInterface(getContextURL(),getApplicationURL(), & mui_);
  xgi::bind(this, &SiStripHistoricInfoClient::handleWebRequest, "Request");
  firstTime = true;
  pSummary_ = new SiStripPerformanceSummary();
  // configure the database
 /*
  db_ = new SiStripConfigDb( pset.getUntrackedParameter<string>("User",""),
				 pset.getUntrackedParameter<string>("Passwd",""),
				 pset.getUntrackedParameter<string>("Path",""),
				 pset.getUntrackedParameter<string>("Partition",""),
				 pset.getUntrackedParameter<unsigned int>("MajorVersion",0),
				 pset.getUntrackedParameter<unsigned int>("MinorVersion",0) );
*/
  //GB: 25/11/07 commented out this apparently useless Online DB dependency
  //  db_ = new SiStripConfigDb( "dorian", "test", "sqlite_file:HistoricPerformance.db", "TEST_PARTITION",0,0);
}


//-----------------------------------------------------------------------------------------------
/*
  implement the method that outputs the page with the widgets (declared in DQMBaseClient):
*/
void SiStripHistoricInfoClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception) {
  // the web interface should know what to do:
  webInterface_p->Default(in, out);
}


//-----------------------------------------------------------------------------------------------
/* the method called on all HTTP requests of the form ".../Request?RequestID=..." */
void SiStripHistoricInfoClient::handleWebRequest(xgi::Input * in, xgi::Output * out) {
  // the web interface should know what to do:
  webInterface_p->handleRequest(in, out);
}


//-----------------------------------------------------------------------------------------------
/* this obligatory method is called whenever the client enters the "Configured" state: */
void SiStripHistoricInfoClient::configure() {
}


//-----------------------------------------------------------------------------------------------
/* this obligatory method is called whenever the client enters the "Enabled" state: */
void SiStripHistoricInfoClient::newRun() {
  upd_->registerObserver(this);   // upd_ is a pointer to dqm::Updater, protected data member of DQMBaseClient
}


//-----------------------------------------------------------------------------------------------
//  this obligatory method is called whenever the client enters the "Halted" state:
void SiStripHistoricInfoClient::endRun(){
  cout<<"SiStripHistoricInfoClient::endRun() : called"<<endl;
//  printMEs();
  retrievePointersToModuleMEs();
  fillSummaryObjects();
  pSummary_->print();
  std::string final_filename = "endRun_SiStripHistoricInfoClient.root"; // run specific filename would be better
  std::cout<<"Saving all histograms in "<<final_filename<<std::endl;
  mui_->getBEInterface()->save(final_filename);
  std::cout<<"Writing objects to DB"<<std::endl;
  writeToDB();
}


//-----------------------------------------------------------------------------------------------
/* this obligatory method is called by the Updater component, whenever there is an update */
void SiStripHistoricInfoClient::onUpdate() const{
  //
  int nr_updates = mui_->getNumUpdates();
  std::cout<<" onUpdate = "<<nr_updates<<std::endl;
  retrievePointersToModuleMEs();

  //
  if(firstTime){
    firstUpdate = nr_updates;
    cout<<"SiStripHistoricInfoClient::onUpdate() first time call. Subscribing. firstUpdate="<<firstUpdate<<endl;
    //    mui_->subscribe("Collector/*/SiStrip/*"); // NOT Supported in DQMServices V3
    firstTime = false; // done, set flag to false again
  }

  // perform ROOT thread-unsafe actions in onUpdate
  if(webInterface_p->getSaveToFile()){
    cout<<"SiStripHistoricInfoClient::onUpdate(). Saving to file."<<endl;
    mui_->getBEInterface()->save("SiStripHistoricInfoClient.root");
    webInterface_p->setSaveToFile(false); // done, set flag to false again
  }

  //
  if( nr_updates == 10 ){
    cout<<"SiStripHistoricInfoClient::onUpdate() : nr_updates="<<nr_updates<<" "<<nr_updates-firstUpdate<<endl;
    retrievePointersToModuleMEs();
    fillSummaryObjects();
    pSummary_->print();
  }
}


//-----------------------------------------------------------------------------------------------
void SiStripHistoricInfoClient::retrievePointersToModuleMEs() const{
// painful and dangerous string operations to extract list of pointer to MEs and avoid strings with full paths
// uses the DQMOldReceiver and fills the data member map
  vector<string> listOfMEsWithFullPath;
  mui_->getBEInterface()->getContents(listOfMEsWithFullPath); // put list of MEs in vector which is passed as parameter to method
  cout<<"SiStripHistoricInfoClient::retrievePointersToModuleMEs : listOfMEsWithFullPath.size() "<<listOfMEsWithFullPath.size()<<endl;
  for(vector<string>::const_iterator ime = listOfMEsWithFullPath.begin(); ime != listOfMEsWithFullPath.end(); ime++){ // loop over strings
     // divide path and histogram names
     uint32_t length_path=(*ime).find(":",0);
     if(length_path==string::npos) continue; // no ":" found, skip this and continue with next iterator step
     string thepath=(*ime).substr(0,length_path); // path part of the string is ended with ":"
     string allhistonames=(*ime).substr(length_path+1); // rest of string is the histogram names
     uint while_counter = 0;
     while(true){                  // implicit loop, go out when no more ','-s are found
       while_counter++;
       uint thehistonamelength = allhistonames.find(",",0);
       string thehistoname;
       if(thehistonamelength != string::npos){
         thehistoname = allhistonames.substr(0,thehistonamelength);
         allhistonames.erase(0,thehistonamelength+1);
       }else{
         thehistoname = allhistonames; // no more ","-s, take all
       }
       string fullhistopath = thepath + "/" + thehistoname;
       // get pointer to each ME
//DKC
       MonitorElement * theMEPointer = mui_->getBEInterface()->get(fullhistopath); // give the full path and get back the pointer to the ME
       // extract detid from id/title - use SistripHistoId for doing this
       SiStripHistoId hidmanager; string histoid="#"; uint32_t theMEDetId = 0;
       if(theMEPointer){
         histoid = theMEPointer->getName(); // get id of histogram
         theMEDetId = hidmanager.getComponentId(histoid);
         std::map<uint32_t , vector<MonitorElement *> >::iterator is_me_in_map = ClientPointersToModuleMEs.find(theMEDetId); // search key in map
         if( is_me_in_map == ClientPointersToModuleMEs.end() ){ // this key is not in map, create new pair and insert it in map
            vector<MonitorElement*> newvec;
            newvec.push_back(theMEPointer);
            ClientPointersToModuleMEs.insert( std::make_pair(theMEDetId,newvec) ); // fill couple in map
         }else{ // this key is already in map, add the ME pointer to its vector
            (is_me_in_map->second).push_back(theMEPointer);
         }
       }
       if(thehistonamelength == string::npos) break; // if no further "," left leave loop
       if(while_counter>15){
         cout<<"SiStripHistoricInfoClient::retrievePointersToModuleMEs while_counter="<<while_counter<<" leaving loop, check this in the code"<<endl;
         break; // limit maximum nr. to 15 just in case it goes crazy and we have an endless loop
       }
     }
  }
}


//-----------------------------------------------------------------------------------------------
//  this obligatory method is called whenever the client enters the "Halted" state:
void SiStripHistoricInfoClient::printMEs() const {
  cout<<"SiStripHistoricInfoClient::printMEs() : called"<<endl;
  cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
  cout<<"SiStripHistoricInfoClient::printMEs ClientPointersToModuleMEs.size()="<<ClientPointersToModuleMEs.size()<<endl;
  for(std::map<uint32_t , vector<MonitorElement *> >::iterator imapmes = ClientPointersToModuleMEs.begin(); imapmes != ClientPointersToModuleMEs.end(); imapmes++){
     cout<<"      ++++++detid  "<<imapmes->first<<endl;
     vector<MonitorElement*> locvec = imapmes->second;
     for(vector<MonitorElement*>::const_iterator imep = locvec.begin(); imep != locvec.end() ; imep++){
       cout<<"          ++  "<<(*imep)->getName()<<" entries/mean/rms : "<<(*imep)->getEntries()<<" / "<<(*imep)->getMean()<<" / "<<(*imep)->getRMS()<<endl;
     }
/*
     DQMStore * dbe_ = mui_->getBEInterface();
     vector<MonitorElement*> tagged_mes = dbe_->get(imapmes->first);
     for(vector<MonitorElement*>::const_iterator imep = tagged_mes.begin(); imep != tagged_mes.end() ; imep++){
       cout<<"tagged_mes++  "<<(*imep)->getName()<<" entries/mean/rms : "<<(*imep)->getEntries()<<" / "<<(*imep)->getMean()<<" / "<<(*imep)->getRMS()<<endl;
     }
*/
  }
  cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
}


//-----------------------------------------------------------------------------------------------
void SiStripHistoricInfoClient::fillSummaryObjects() const {
  cout<<"SiStripHistoricInfoClient::fillSummaryObjects() : called"<<endl;
  for(std::map<uint32_t , vector<MonitorElement *> >::iterator imapmes = ClientPointersToModuleMEs.begin(); imapmes != ClientPointersToModuleMEs.end(); imapmes++){
     uint32_t local_detid = imapmes->first;
     vector<MonitorElement*> locvec = imapmes->second;
     for(vector<MonitorElement*>::const_iterator imep = locvec.begin(); imep != locvec.end() ; imep++){
       std::string MEName = (*imep)->getName(); unsigned int ipos = MEName.find("ClusterCharge__");
       if(ipos != std::string::npos){
         std::vector<float> summary_values;
         summary_values.push_back((*imep)->getEntries());
         summary_values.push_back((*imep)->getMean());
         summary_values.push_back((*imep)->getRMS());
         pSummary_->setDet(local_detid, summary_values);
       }
     }
  }
}


//-----------------------------------------------------------------------------------------------
void SiStripHistoricInfoClient::writeToDB() const {
/* // old TStore stuff
   TSQLServer  *dbserver = TSQLServer::Connect("oracle://devb10","CMS_TRACKER_GBRUNO","client4histoplot");
   TSQLResult* res = dbserver->Query( "SELECT * FROM PEDESTALS");
   delete res;
   delete dbserver;
*/
  bool appendMode_ = false;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( mydbservice.isAvailable() ){
    unsigned long long tillTime;
    if ( appendMode_){
      tillTime = mydbservice->currentTime();
    } else {
      tillTime = mydbservice->endOfTime();
    }
    std::cout<<"SiStripHistoricInfoClient::writeToDB tillTime = " << tillTime << std::endl;
    
    if( mydbservice->isNewTagRequest("SiStripPerformanceSummaryRcd") ){
      mydbservice->createNewIOV<SiStripPerformanceSummary>(pSummary_,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripPerformanceSummaryRcd");      
    } else {
      mydbservice->appendSinceTime<SiStripPerformanceSummary>(pSummary_,mydbservice->currentTime(),"SiStripPerformanceSummaryRcd");      
    }
  }else{
    std::cout<<"ERROR SiStripHistoricInfoClient::writeToDB PoolDBOutputService is unavailable"<<std::endl;
  }
}


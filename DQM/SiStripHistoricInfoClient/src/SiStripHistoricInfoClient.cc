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
// $Id: SiStripHistoricInfoClient.cc,v 1.5 2006/12/05 17:46:07 dkcira Exp $
//

#include "DQM/SiStripHistoricInfoClient/interface/SiStripHistoricInfoClient.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

#include <vector>
#include <string>
#include <iostream>
#include<sstream>



#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/Method.h"
#include "xoap/SOAPEnvelope.h"
#define TSTORE_NS_URI "http://xdaq.web.cern.ch/xdaq/xsd/2006/tstore-10.xsd" //eventually I suppose this will be defined in a header somewhere


using namespace std;
using namespace cgicc;
using namespace xcept;

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
}

/*
  implement the method that outputs the page with the widgets (declared in DQMBaseClient):
*/
void SiStripHistoricInfoClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  // the web interface should know what to do:
  webInterface_p->Default(in, out);
}


/*
  the method called on all HTTP requests of the form ".../Request?RequestID=..."
*/
void SiStripHistoricInfoClient::handleWebRequest(xgi::Input * in, xgi::Output * out)
{
  // the web interface should know what to do:
  webInterface_p->handleRequest(in, out);
}

/*
  this obligatory method is called whenever the client enters the "Configured" state:
*/
void SiStripHistoricInfoClient::configure()
{

}

/*
  this obligatory method is called whenever the client enters the "Enabled" state:
*/
void SiStripHistoricInfoClient::newRun()
{
  upd_->registerObserver(this);   // upd_ is a pointer to dqm::Updater, protected data member of DQMBaseClient
}

//  this obligatory method is called whenever the client enters the "Halted" state:
void SiStripHistoricInfoClient::endRun()
{
  cout<<"SiStripHistoricInfoClient::endRun() : called"<<endl;
  cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
  cout<<"SiStripHistoricInfoClient::endRun ClientPointersToModuleMEs.size()="<<ClientPointersToModuleMEs.size()<<endl;
  for(std::map<uint32_t , vector<MonitorElement *> >::iterator imapmes = ClientPointersToModuleMEs.begin(); imapmes != ClientPointersToModuleMEs.end(); imapmes++){
     cout<<"      ++++++detid  "<<imapmes->first<<endl;
     vector<MonitorElement*> locvec = imapmes->second;
     DaqMonitorBEInterface * dbe_ = mui_->getBEInterface();
     vector<MonitorElement*> tagged_mes = dbe_->get(imapmes->first);
     for(vector<MonitorElement*>::const_iterator imep = locvec.begin(); imep != locvec.end() ; imep++){
       cout<<"          ++  "<<(*imep)->getName()<<" entries/mean/rms : "<<(*imep)->getEntries()<<" / "<<(*imep)->getMean()<<" / "<<(*imep)->getRMS()<<endl;
     }
     for(vector<MonitorElement*>::const_iterator imep = tagged_mes.begin(); imep != tagged_mes.end() ; imep++){
       cout<<"tagged_mes++  "<<(*imep)->getName()<<" entries/mean/rms : "<<(*imep)->getEntries()<<" / "<<(*imep)->getMean()<<" / "<<(*imep)->getRMS()<<endl;
     }
  }
  cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
  std::string final_filename = "endRun_SiStripHistoricInfoClient.root"; // run specific filename would be better
  std::cout<<"Saving all histograms in "<<final_filename<<std::endl;
  mui_->save(final_filename);

//  tstore_connect();
}

/*
  this obligatory method is called by the Updater component, whenever there is an update 
*/
void SiStripHistoricInfoClient::onUpdate() const
{
  //
  if(firstTime){
    firstUpdate = mui_->getNumUpdates();
    cout<<"SiStripHistoricInfoClient::onUpdate() first time call. Subscribing. firstUpdate="<<firstUpdate<<endl;
    mui_->subscribe("Collector/*/SiStrip/*");
    firstTime = false; // done, set flag to false again
  }

  // perform ROOT thread-unsafe actions in onUpdate
  if(webInterface_p->getSaveToFile()){
    cout<<"SiStripHistoricInfoClient::onUpdate(). Saving to file."<<endl;
    mui_->save("SiStripHistoricInfoClient.root");
    webInterface_p->setSaveToFile(false); // done, set flag to false again
  }

  //
  int nr_updates = mui_->getNumUpdates();
  cout<<"SiStripHistoricInfoClient::onUpdate() : nr_updates="<<nr_updates<<" "<<nr_updates-firstUpdate<<endl;
  if(nr_updates==2){
    retrievePointersToModuleMEs();
    cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
    cout<<"SiStripHistoricInfoClient::retrievePointersToModuleMEs ClientPointersToModuleMEs.size()="<<ClientPointersToModuleMEs.size()<<endl;
    for(std::map<uint32_t , vector<MonitorElement *> >::iterator imapmes = ClientPointersToModuleMEs.begin(); imapmes != ClientPointersToModuleMEs.end(); imapmes++){
       cout<<"      ++++++detid  "<<imapmes->first<<endl;
       // MEs from pointer map
       vector<MonitorElement*> locvec = imapmes->second;
       for(vector<MonitorElement*>::const_iterator imep = locvec.begin(); imep != locvec.end() ; imep++){
         cout<<"          ++  "<<(*imep)->getName()<<endl;
       }
       //  tagged MEs
       DaqMonitorBEInterface * dbe_ = mui_->getBEInterface();
       std::vector<MonitorElement *> taggedMEs = dbe_->get(imapmes->first);
       for(std::vector<MonitorElement *>::const_iterator itme = taggedMEs.begin(); itme != taggedMEs.end(); itme++){
         cout<<"          --  "<<(*itme)->getName()<<endl;
       }
    }
    cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
  }
}

void SiStripHistoricInfoClient::fillSummaryObjects() {
//   map<uint32_t, pair<double, double>> ClusterChargeMeanRMS;
//   map<uint32_t, pair<double, double>> OccupancyMeanRMS;
}

void SiStripHistoricInfoClient::retrievePointersToModuleMEs() const{
// painful and dangerous string operations to extract list of pointer to MEs and avoid strings with full paths
// uses the MonitorUserInterface and fills the data member map
  vector<string> listOfMEsWithFullPath;
  mui_->getContents(listOfMEsWithFullPath); // put list of MEs in vector which is passed as parameter to method
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
       MonitorElement * theMEPointer = mui_->get(fullhistopath); // give the full path and get back the pointer to the ME
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
//  cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
  cout<<"SiStripHistoricInfoClient::retrievePointersToModuleMEs ClientPointersToModuleMEs.size()="<<ClientPointersToModuleMEs.size()<<endl;
//  for(std::map<uint32_t , vector<MonitorElement *> >::iterator imapmes = ClientPointersToModuleMEs.begin(); imapmes != ClientPointersToModuleMEs.end(); imapmes++){
//     cout<<"      ++++++detid  "<<imapmes->first<<endl;
//     vector<MonitorElement*> locvec = imapmes->second;
//     for(vector<MonitorElement*>::const_iterator imep = locvec.begin(); imep != locvec.end() ; imep++){
//       cout<<"          ++  "<<(*imep)->getName()<<endl;
//     }
//  }
//  cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
}


void SiStripHistoricInfoClient::tstore_connect(){
  cout<<"SiStripHistoricInfoClient::tstore_connect()  called"<<endl;
  xoap::MessageReference msg = xoap::createMessage();
  try {
        xoap::SOAPEnvelope envelope = msg->getSOAPPart().getEnvelope();
        xoap::SOAPName msgName = envelope.createName( "connect", "tstore", "http://xdaq.web.cern.ch/xdaq/xsd/2006/tstore-10.xsd");
        xoap::SOAPElement connectElement = envelope.getBody().addBodyElement ( msgName );

        xoap::SOAPName id = envelope.createName("id", "tstore", "http://xdaq.web.cern.ch/xdaq/xsd/2006/tstore-10.xsd");
        connectElement.addAttribute(id, "urn:tstore-view-SQL:MyParameterisedView");
        xoap::SOAPName passwordName = envelope.createName("password", "tstore", "http://xdaq.web.cern.ch/xdaq/xsd/2006/tstore-10.xsd");
        connectElement.addAttribute(passwordName, "grape");
  }catch(xoap::exception::Exception& e) {
   //handle exception
  }
}

#define TSTORE_NS_URI "http://xdaq.web.cern.ch/xdaq/xsd/2006/tstore-10.xsd" 
// will be defined in another header file

#include <vector>
#include <string>
#include <iostream>
#include <sstream>

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/Method.h"
#include "xoap/SOAPEnvelope.h"

#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoClient.h"

using namespace std;
using namespace cgicc;
using namespace xcept;


SiPixelHistoricInfoClient::SiPixelHistoricInfoClient(xdaq::ApplicationStub* stub) 
  			 : DQMBaseClient(stub,       // application stub - never change it 
   			 		"test",      // name for collector to identify client
			 		"localhost", // name of the collector host 
			 		 9090) {     // port the collector listens to 
  // instantiate a web interface:
  webInterface_p = new SiPixelHistoricInfoWebInterface(getContextURL(), getApplicationURL(), &mui_);
  xgi::bind(this, &SiPixelHistoricInfoClient::handleWebRequest, "Request");
}


// to output webpages with the widgets declared in DQMBaseClient
void SiPixelHistoricInfoClient::general(xgi::Input* in, xgi::Output* out) 
				throw (xgi::exception::Exception) { 
  webInterface_p->Default(in, out); // web interface should know what to do 
}


// to call on http-request of form ".../Request?RequestID=..."
void SiPixelHistoricInfoClient::handleWebRequest(xgi::Input* in, xgi::Output* out) {
  webInterface_p->handleRequest(in, out); // web interface should know what to do 
}

// to call obligatorily when the client enters the "Configured" state
void SiPixelHistoricInfoClient::configure() {}


// to call obligatorily when the client enters the "Enabled" state
void SiPixelHistoricInfoClient::newRun() {
  upd_->registerObserver(this); 
  // upd_ is a pointer to dqm::Updater, protected data member of DQMBaseClient
}


// to call obligatorily when the client enters the "Halted" state 
void SiPixelHistoricInfoClient::endRun() {
  cout << "SiPixelHistoricInfoClient::endRun() called" << endl;
  cout << "+++++++++++++++++++++++++++++++++" << endl;
  cout << "SiPixelHistoricInfoClient::endRun ClientPointersToModuleMEs.size() = "
       <<  ClientPointersToModuleMEs.size() << endl;

  for (std::map< uint32_t, vector<MonitorElement*> >::iterator imapmes=ClientPointersToModuleMEs.begin(); 
       imapmes!=ClientPointersToModuleMEs.end(); imapmes++) {
     cout << "detid = " << imapmes->first << endl;
     vector<MonitorElement*> locvec = imapmes->second;
     DaqMonitorBEInterface* dbe_ = mui_->getBEInterface();
     vector<MonitorElement*> tagged_mes = dbe_->get(imapmes->first);
     for (vector<MonitorElement*>::const_iterator imep=locvec.begin(); imep!=locvec.end(); imep++) {
       cout << (*imep)->getName() << " entries/mean/rms: " << (*imep)->getEntries() << "/" 
                                                           << (*imep)->getMean() << "/" 
							   << (*imep)->getRMS() << endl;
     }
     for (vector<MonitorElement*>::const_iterator imep=tagged_mes.begin(); imep!=tagged_mes.end(); imep++) {
       cout << "tagged_mes " << (*imep)->getName() << " entries/mean/rms: " << (*imep)->getEntries() << "/" 
                                                                            << (*imep)->getMean() << "/" 
									    << (*imep)->getRMS() << endl;
     }
  }
  cout << "+++++++++++++++++++++++++++++++++" << endl;
  std::string final_filename = "endRun_SiPixelHistoricInfoClient.root"; // run-specific filename would be better
  std::cout << "saving all histograms to " << final_filename << std::endl;
  mui_->save(final_filename); 
  
  tstore_connect();
}


// to call obligatorily by the Updater component when there is update 
void SiPixelHistoricInfoClient::onUpdate() const {
  if (firstTime) {
    firstUpdate = mui_->getNumUpdates();
    cout << "SiPixelHistoricInfoClient::onUpdate() first time call. Subscribing. firstUpdate = " << firstUpdate << endl;
    mui_->subscribe("Collector/*/SiPixel/*");
    firstTime = false; // done, set flag to false again
  }
  // perform ROOT thread-unsafe actions in onUpdate
  if (webInterface_p->getSaveToFile()) {
    cout << "SiPixelHistoricInfoClient::onUpdate(). Saving to file." << endl;
    mui_->save("SiPixelHistoricInfoClient.root");
    webInterface_p->setSaveToFile(false); // done, set flag to false again
  }
  int nr_updates = mui_->getNumUpdates();
  cout << "SiPixelHistoricInfoClient::onUpdate(): nr_updates = " << nr_updates << " " << nr_updates-firstUpdate << endl;
  if (nr_updates==2) {
    retrievePointersToModuleMEs();
    cout << "+++++++++++++++++++++++++++++++++" << endl;
    cout << "SiPixelHistoricInfoClient::retrievePointersToModuleMEs ClientPointersToModuleMEs.size() = " 
         << ClientPointersToModuleMEs.size() << endl;
    for (std::map< uint32_t, vector<MonitorElement*> >::iterator imapmes=ClientPointersToModuleMEs.begin(); 
         imapmes!=ClientPointersToModuleMEs.end(); imapmes++) {
      cout << "detid = " << imapmes->first << endl;
      vector<MonitorElement*> locvec = imapmes->second; // MEs from pointer map
      for (vector<MonitorElement*>::const_iterator imep=locvec.begin(); imep!=locvec.end(); imep++) {
        cout << (*imep)->getName() << endl;
      }
      DaqMonitorBEInterface* dbe_ = mui_->getBEInterface();
      std::vector<MonitorElement*> taggedMEs = dbe_->get(imapmes->first);
      for (std::vector<MonitorElement*>::const_iterator itme=taggedMEs.begin(); itme!=taggedMEs.end(); itme++) {
        cout << (*itme)->getName() << endl;
      }
    }
    cout << "+++++++++++++++++++++++++++++++++" << endl;
  }
}


void SiPixelHistoricInfoClient::fillSummaryObjects() {
  // map< uint32_t, pair<double, double> > ClusterChargeMeanRMS;
  // map< uint32_t, pair<double, double> > OccupancyMeanRMS;
}


void SiPixelHistoricInfoClient::retrievePointersToModuleMEs() const {
  // painful and dangerous string operations to extract list of pointer to MEs and avoid 
  // strings with full paths uses the MonitorUserInterface and fills the data member map
  vector<string> listOfMEsWithFullPath;
  mui_->getContents(listOfMEsWithFullPath); 
  // put the list of MEs in a vector to pass as a parameter to methods
  
  cout << "SiPixelHistoricInfoClient::retrievePointersToModuleMEs: listOfMEsWithFullPath.size() = " 
       << listOfMEsWithFullPath.size() << endl;

  for (vector<string>::const_iterator ime=listOfMEsWithFullPath.begin(); 
      ime!=listOfMEsWithFullPath.end(); ime++) { // loop over strings
    // divide path and histogram names
    uint32_t length_path = (*ime).find(":", 0);
    if (length_path==string::npos) continue; // no ":" found, skip this and continue with next iterator step
    
    string thepath = (*ime).substr(0, length_path); // path part of the string is ended with ":"
    string allhistonames = (*ime).substr(length_path+1); // rest of string is the histogram names
    uint while_counter=0;
    while (true) { // implicit loop, go out when no more ','-s are found
      while_counter++;
      uint thehistonamelength = allhistonames.find(",", 0);
      string thehistoname;
      if (thehistonamelength!=string::npos) {
    	thehistoname = allhistonames.substr(0, thehistonamelength);
    	allhistonames.erase(0, thehistonamelength+1);
      }
      else { 
    	thehistoname = allhistonames; // no more ","-s, take all
      }
      string fullhistopath = thepath + "/" + thehistoname;

      // get a pointer to each ME
      MonitorElement* theMEPointer = mui_->get(fullhistopath); // give the full path and get back the pointer to the ME

      // extract detid from id/title - use SiPixelHistogramId for doing this
      SiPixelHistogramId hIdManager; 
      string histoId="#"; 
      uint32_t theMEDetId=0;
      if (theMEPointer) {
    	histoId = theMEPointer->getName(); 
    	theMEDetId = hIdManager.getRawId(histoId);
	// search key in map
    	std::map< uint32_t, vector<MonitorElement*> >::iterator is_me_in_map = ClientPointersToModuleMEs.find(theMEDetId); 
    	if (is_me_in_map==ClientPointersToModuleMEs.end()) { 
          // if this key is not in map, create new pair and insert it in map
    	  vector<MonitorElement*> newvec;
    	  newvec.push_back(theMEPointer);
    	  ClientPointersToModuleMEs.insert(std::make_pair(theMEDetId, newvec)); // fill couple in map
    	}
        else { 
          // if the key is already in map, add the ME pointer to its vector
    	  (is_me_in_map->second).push_back(theMEPointer);
    	}
      }
      if (thehistonamelength==string::npos) break; // if no further "," left leave loop
      if (while_counter>15) {
    	cout << "SiPixelHistoricInfoClient::retrievePointersToModuleMEs while_counter = " 
             <<  while_counter << " leaving loop, check this in the codes" << endl;
    	break; // limit maximum nr to 15 just in case it goes crazy and we have an endless loop
      }
    }
  }
  cout << "SiPixelHistoricInfoClient::retrievePointersToModuleMEs ClientPointersToModuleMEs.size() = "
       <<  ClientPointersToModuleMEs.size() << endl;
}


void SiPixelHistoricInfoClient::tstore_connect () {
  cout << "SiPixelHistoricInfoClient::tstore_connect() called" << endl;
  xoap::MessageReference msg = xoap::createMessage(); 
  try {
    xoap::SOAPEnvelope envelope = msg->getSOAPPart().getEnvelope();
    xoap::SOAPName msgName = envelope.createName("connect", "tstore", "http://xdaq.web.cern.ch/xdaq/xsd/2006/tstore-10.xsd");
    xoap::SOAPElement connectElement = envelope.getBody().addBodyElement ( msgName );

    xoap::SOAPName id = envelope.createName("id", "tstore", "http://xdaq.web.cern.ch/xdaq/xsd/2006/tstore-10.xsd");
    connectElement.addAttribute(id, "urn:tstore-view-SQL:MyParameterisedView");
    xoap::SOAPName passwordName = envelope.createName("password", "tstore", "http://xdaq.web.cern.ch/xdaq/xsd/2006/tstore-10.xsd");
    connectElement.addAttribute(passwordName, "grape");
  }
  catch (xoap::exception::Exception& e) { /* handle exception */ }
}

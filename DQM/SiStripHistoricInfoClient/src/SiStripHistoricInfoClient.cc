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
// $Id: SiStripHistoricInfoClient.cc,v 1.11 2006/12/13 16:26:04 dkcira Exp $
//

#include "DQM/SiStripHistoricInfoClient/interface/SiStripHistoricInfoClient.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

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

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/Method.h"
#include "xoap/domutils.h"
#include "xoap/SOAPElement.h"
#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xercesc/dom/DOMNode.hpp"

#include "toolbox/net/Utils.h"

#include "tstore/tstore/include/AttachmentUtils.h"

#define TSTORE_NS_URI "http://xdaq.web.cern.ch/xdaq/xsd/2006/tstore-10.xsd" //eventually I suppose this will be defined in a header somewhere

using namespace std;
using namespace cgicc;
using namespace xcept;


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
}


//-----------------------------------------------------------------------------------------------
/*
  implement the method that outputs the page with the widgets (declared in DQMBaseClient):
*/
void SiStripHistoricInfoClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  // the web interface should know what to do:
  webInterface_p->Default(in, out);
}


//-----------------------------------------------------------------------------------------------
/* the method called on all HTTP requests of the form ".../Request?RequestID=..." */
void SiStripHistoricInfoClient::handleWebRequest(xgi::Input * in, xgi::Output * out)
{
  // the web interface should know what to do:
  webInterface_p->handleRequest(in, out);
}


//-----------------------------------------------------------------------------------------------
/* this obligatory method is called whenever the client enters the "Configured" state: */
void SiStripHistoricInfoClient::configure()
{
  tstore_sistrip();
}


//-----------------------------------------------------------------------------------------------
/* this obligatory method is called whenever the client enters the "Enabled" state: */
void SiStripHistoricInfoClient::newRun()
{
  upd_->registerObserver(this);   // upd_ is a pointer to dqm::Updater, protected data member of DQMBaseClient
}


//-----------------------------------------------------------------------------------------------
//  this obligatory method is called whenever the client enters the "Halted" state:
void SiStripHistoricInfoClient::endRun()
{
/*
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
*/
//  std::string final_filename = "endRun_SiStripHistoricInfoClient.root"; // run specific filename would be better
//  std::cout<<"Saving all histograms in "<<final_filename<<std::endl;
//  mui_->save(final_filename);
}


//-----------------------------------------------------------------------------------------------
/* this obligatory method is called by the Updater component, whenever there is an update */
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
  if(nr_updates==4){
    cout<<"SiStripHistoricInfoClient::onUpdate() : retrieving pointers to histograms"<<std::endl;
    retrievePointersToModuleMEs();
/*
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
*/
  }
}


//-----------------------------------------------------------------------------------------------
void SiStripHistoricInfoClient::fillSummaryObjects() {
//   map<uint32_t, pair<double, double>> ClusterChargeMeanRMS;
//   map<uint32_t, pair<double, double>> OccupancyMeanRMS;
}


//-----------------------------------------------------------------------------------------------
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
}


//-----------------------------------------------------------------------------------------------
void SiStripHistoricInfoClient::tstore_sistrip(){
  cout<<"SiStripHistoricInfoClient::tstore_sistrip()  called"<<endl;

  // create message
  xoap::MessageReference msg1 = xoap::createMessage();
  try {
        xoap::SOAPEnvelope envelope = msg1->getSOAPPart().getEnvelope();
        xoap::SOAPName msgName = envelope.createName( "connect", "sistripview", TSTORE_NS_URI);
        xoap::SOAPElement connectElement = envelope.getBody().addBodyElement ( msgName );
        xoap::SOAPName id = envelope.createName("id", "sistripview", TSTORE_NS_URI);
        connectElement.addAttribute(id, "urn:tstore-view-SQL:sistripview");
        xoap::SOAPName passwordName = envelope.createName("password", "sistripview", TSTORE_NS_URI);
        connectElement.addAttribute(passwordName, "client4histoplot");
        std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- created envelope"<<std::endl;
  }catch(xoap::exception::Exception& e) {
   //handle exception
	std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- xoap::exception"<<std::endl;
  }

  // send message to TStore and get back the id of the connection
  std::string connectionID; // keep this definition out of try-catch since you need it later
  try {
    xdaq::ApplicationDescriptor * tstoreDescriptor = getApplicationContext()->getDefaultZone()->getApplicationDescriptor(getApplicationContext()->getContextDescriptor(),120);
    xoap::MessageReference reply = getApplicationContext()->postSOAP(msg1, tstoreDescriptor);
    xoap::SOAPBody body = reply->getSOAPPart().getEnvelope().getBody();
    if (body.hasFault()) {
      //connection could not be opened
      std::cout<<"SiStripHistoricInfoClient::tstore_sistrip -- connection could NOT be opened"<<std::endl;
    } else {
      DOMNode *connectResponse=  SiStripHistoricInfoClient::getNodeNamed(reply,"connectResponse");
      //store connectionID somewhere so that it can be used for other messages
      connectionID=xoap::getNodeAttribute(connectResponse,"connectionID");
      std::cout<<"SiStripHistoricInfoClient::tstore_sistrip -- connectionID = "<<connectionID<<std::endl;
    }
  } catch (xdaq::exception::Exception& e) {
    std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- connection exception "<<e.what()<<std::endl;
  }

  // TStore retrieval of a data table definition
  xdata::Table sistriptable1;
  { // just for having separate namespace
  xoap::MessageReference msg2 = xoap::createMessage();
    xoap::SOAPEnvelope envelope = msg2->getSOAPPart().getEnvelope();
    xoap::SOAPName msgName = envelope.createName( "definition", "sistripview", TSTORE_NS_URI);
    xoap::SOAPElement element=envelope.getBody().addBodyElement ( msgName );
    //connectionID was obtained from a previous connect message
    xoap::SOAPName id = envelope.createName("connectionID", "sistripview", TSTORE_NS_URI);
    element.addAttribute(id, connectionID);
    //add parameters to the message (for an SQLView, this is just the "name" parameter to specify which insert configuration to use.)
    element.addNamespaceDeclaration("sql",  "urn:tstore-view-SQL");
    xoap::SOAPName property = envelope.createName("name", "sql","urn:tstore-view-SQL");
    element.addAttribute(property, "SISTRIPHISTORICINFOTABLE");
    xdaq::ApplicationDescriptor * tstoreDescriptor = getApplicationContext()->getDefaultZone()->getApplicationDescriptor(getApplicationContext()->getContextDescriptor(),120);
    try{
    xoap::MessageReference reply = getApplicationContext()->postSOAP(msg2, tstoreDescriptor);
    xoap::SOAPBody body = reply->getSOAPPart().getEnvelope().getBody();
      if (!body.hasFault()) {
         std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- body is ok"<<std::endl;
         sistriptable1 = ExtractTableFromAttachment(reply);
      }else{
         std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- body fault "<<body.getFault().getFaultString()<<std::endl;
      }
    } catch(xcept::Exception &e) {
      std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- definition exception "<<e.what()<<std::endl;
    }
  }

  // add rows to table
    std::srand (time(NULL));
  unsigned int rowIndex=sistriptable1.getRowCount();
  std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- rowIndex "<<rowIndex<<std::endl;
  const std::string columnIterator1 = "IOV_VALUE_ID"; const std::string columnIterator2 = "TILLTIME";
  std::vector<std::string> columns=sistriptable1.getColumns();
  for(vector<std::string>::iterator columnIterator = columns.begin(); columnIterator != columns.end(); columnIterator++){
     xdata::Serializable * xdataValue=new xdata::UnsignedLong( (unsigned long)std::rand() );
     std::string columnType=sistriptable1.getColumnType(*columnIterator);
     std::cout<<"SiStripHistoricInfoClient::tstore_sistrip -- inserting "<<xdataValue<<" "<<xdataValue->toString()<<" into column "<<*columnIterator<<" of type "<<columnType<<std::endl;
     sistriptable1.setValueAt(rowIndex,*columnIterator,*xdataValue);
      delete xdataValue;
  }

  // insert back in DB
  {
    xoap::MessageReference msg3 = xoap::createMessage();
    xoap::SOAPEnvelope envelope = msg3->getSOAPPart().getEnvelope();
    xoap::SOAPName msgName = envelope.createName( "insert", "sistripview", TSTORE_NS_URI);
    xoap::SOAPElement element = envelope.getBody().addBodyElement ( msgName );
    //connectionID was obtained from a previous connect message
    xoap::SOAPName id = envelope.createName("connectionID", "sistripview", TSTORE_NS_URI);
    element.addAttribute(id, connectionID);
    //add parameters to the message (for an SQLView, this is just the "name" parameter to specify which insert configuration to use.)
    element.addNamespaceDeclaration("sql",  "urn:tstore-view-SQL");
    xoap::SOAPName property = envelope.createName("name", "sql","urn:tstore-view-SQL");
    element.addAttribute(property, "siStripHistoricInfoTable");
    xdata::exdr::AutoSizeOutputStreamBuffer outBuffer;
    xdata::exdr::Serializer serializer;
    serializer.exportAll( &sistriptable1, &outBuffer );
    xoap::AttachmentPart * attachment = msg3->createAttachmentPart(outBuffer.getBuffer(), outBuffer.tellp(), "application/xdata+table");
    attachment->setContentEncoding("exdr");
    std::string contentId="siStripHistoricInfoTable";
    attachment->setContentId(contentId);
    msg3->addAttachmentPart(attachment);
    xdaq::ApplicationDescriptor * tstoreDescriptor = getApplicationContext()->getDefaultZone()->getApplicationDescriptor(getApplicationContext()->getContextDescriptor(),120);
    try {
      xoap::MessageReference reply = getApplicationContext()->postSOAP(msg3, tstoreDescriptor);
    } catch (xdaq::exception::Exception& e) {
      std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- insertion exception "<<e.what()<<std::endl;
    }
  }

  // close connection
  try {
    xoap::MessageReference msg4 = xoap::createMessage();
    xoap::SOAPEnvelope envelope = msg4->getSOAPPart().getEnvelope();
    xoap::SOAPName msgName = envelope.createName( "disconnect", "sistripview", TSTORE_NS_URI);
    xoap::SOAPElement queryElement = envelope.getBody().addBodyElement ( msgName );
    xoap::SOAPName id = envelope.createName("connectionID", "tstore", TSTORE_NS_URI);
    queryElement.addAttribute(id, connectionID);   
    xdaq::ApplicationDescriptor * tstoreDescriptor = getApplicationContext()->getDefaultZone()->getApplicationDescriptor(getApplicationContext()->getContextDescriptor(),120);
    xoap::MessageReference reply = getApplicationContext()->postSOAP(msg4, tstoreDescriptor);
    std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- disconnected"<<std::endl;
  } catch(xoap::exception::Exception& e) {
    std::cout<<" SiStripHistoricInfoClient::tstore_sistrip -- disconnection exception "<<e.what()<<std::endl;
  }

}


//-----------------------------------------------------------------------------------------------
DOMNode *SiStripHistoricInfoClient::getNodeNamed(xoap::MessageReference msg,const std::string &nodeName) throw (xcept::Exception) {
        xoap::SOAPEnvelope envelope = msg->getSOAPPart().getEnvelope();
        xoap::SOAPBody body = envelope.getBody();
        DOMNode* node = body.getDOMNode();
        DOMNodeList* bodyList = node->getChildNodes();
        for (unsigned int itemIndex = 0; itemIndex < bodyList->getLength(); itemIndex++) {
                DOMNode* child = bodyList->item(itemIndex);
                if (child->getNodeType() == DOMNode::ELEMENT_NODE) {
                        if (xoap::XMLCh2String(child->getLocalName()) == nodeName) {
                                return child;
                        }
                }
        }
        XCEPT_RAISE(xcept::Exception,"No node named "+nodeName);
}


//-----------------------------------------------------------------------------------------------
xdata::Table SiStripHistoricInfoClient::ExtractTableFromAttachment(xoap::MessageReference reply){
  //  extract the data from the attachment in the reply
  xdata::Table t;
  std::list<xoap::AttachmentPart*> attachments = reply->getAttachments();
  std::cout<<" SiStripHistoricInfoClient::ExtractTableFromAttachment -- attachments.size() "<<attachments.size()<<std::endl;
  std::list<xoap::AttachmentPart*>::iterator j;
  for ( j = attachments.begin(); j != attachments.end(); j++ ) {
        std::cout<<" SiStripHistoricInfoClient::ExtractTableFromAttachment -- next attachment"<<std::endl;
    if ((*j)->getContentType() == "application/xdata+table") {
      xdata::exdr::FixedSizeInputStreamBuffer inBuffer((*j)->getContent(),(*j)->getSize());
      std::string contentEncoding = (*j)->getContentEncoding();
      std::string contentId = (*j)->getContentId();
      std::cout<<" SiStripHistoricInfoClient::ExtractTableFromAttachment -- contentEncoding="<<contentEncoding<<std::endl;
      std::cout<<" SiStripHistoricInfoClient::ExtractTableFromAttachment -- contentId="<<contentId<<std::endl;
      try {
            xdata::exdr::Serializer serializer;
            serializer.import(&t, &inBuffer );
      } catch(xdata::exception::Exception & e ) {
        std::cout<<" SiStripHistoricInfoClient::ExtractTableFromAttachment -- serializer exception ="<<e.what()<<std::endl;
      }
/*`
      // print out table again
      for (xdata::Table::iterator ti = t.begin(); ti != t.end(); ti++) {
          std::cout<<" SiStripHistoricInfoClient::ExtractTableFromAttachment -- next row"<<std::endl;
          xdata::UnsignedLong * number = dynamic_cast<xdata::UnsignedLong *>((*ti).getField("IOV_VALUE_ID"));
          xdata::UnsignedLong * time = dynamic_cast<xdata::UnsignedLong *>((*ti).getField("TILLTIME"));
          std::cout<<"time pointer "<<time<<std::endl;
          std::cout<<"number pointer "<<number<<std::endl;
          std::cout <<" SiStripHistoricInfoClient::ExtractTableFromAttachment -- number / time "<< number->toString() << "\t" << time->toString() << std::endl;
      }
*/
     return t; // return first table found
    }
  }
  return t; // return empty table is nothing was found
}



/** \file
 *
 *  Implementation of  DTDQMClient
 *
 *  $Date: 2006/06/28 11:15:42 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */



#include "DQM/DTMonitorClient/interface/DTDQMClient.h"
#include "DQM/DTMonitorClient/interface/DTNoiseClient.h"

#include "DQMServices/ClientConfig/interface/SubscriptionHandle.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"


DTDQMClient::DTDQMClient(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "test",     // the name by which the collector identifies the client
		  "localhost",// the name of the computer hosting the collector
		  9090        // the port at which the collector listens
		  ),QTFailed(0),QTCritical(0)
{
  webInterface_p = new DTWebInterface(getContextURL(),getApplicationURL(), & mui_);
  
  xgi::bind(this, &DTDQMClient::handleWebRequest, "Request");

  logFile.open("DTDQMClient.log");
  qtestsConfigured=false;
  meListConfigured=false;
	
  subscriber=new SubscriptionHandle;
  qtHandler=new QTestHandle;
  
  noiseClient = new DTNoiseClient;
}



void DTDQMClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception){
  webInterface_p->Default(in, out);
}




void DTDQMClient::handleWebRequest(xgi::Input * in, xgi::Output * out){
  webInterface_p->handleRequest(in, out);
}



void DTDQMClient::configure(){
	
  meListConfigured=false;
  qtestsConfigured=false;

  //#ifdef DT_CLIENT_DEBUG
  //logFile << "Configuring DTDQMClient" << std::endl;
  //#endif
	
  if(!subscriber->getMEList("MESubscriptionList.xml")) meListConfigured=true; 
  if(!qtHandler->configureTests("QualityTests.xml",mui_)) qtestsConfigured=true; 
}

void DTDQMClient::newRun(){
  upd_->registerObserver(this);
	
  ///ME's subscription
  if(meListConfigured) {
    subscriber->makeSubscriptions(mui_);
  }else{
    logFile << "Cannot subscribe to ME's, error occurred in configuration." << std::endl;		
  }
  ///QT's enabling
  if(qtestsConfigured){
    qtHandler->attachTests(mui_);	
  }else{
    logFile << "Cannot run quality tests, error occurred in configuration." << std::endl;		
  }

}

void DTDQMClient::endRun(){

}

void DTDQMClient::onUpdate() const{

  // put here the code that needs to be executed on every update:
  std::vector<std::string> uplist;
  mui_->getUpdatedContents(uplist);
  
  if(meListConfigured) subscriber->updateSubscriptions(mui_);
  if(qtestsConfigured){	
    
    if (webInterface_p->globalQTStatusRequest()) qtHandler->checkGolbalQTStatus(mui_);
    if (webInterface_p->detailedQTStatusRequest()) qtHandler->checkDetailedQTStatus(mui_);

    // My stuff
    if (webInterface_p->noiseStatus()) {

      noiseClient->performCheck(mui_);

    }

    if (!webInterface_p->noiseStatus()) cout<<"[DTDQMClient]: I stopped the noiseCheck"<<endl;
	  
  }

  

}










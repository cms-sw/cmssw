/** \file
 *
 *  Implementation of  L1TClient
 *
 *  $Date: 2007/04/23 15:29:45 $
 *  $Revision: 1.1 $
 *  \author Lorenzo Agostino
 */



#include "DQM/L1TMonitorClient/interface/L1TClient.h"

#include "DQMServices/ClientConfig/interface/SubscriptionHandle.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"


L1TClient::L1TClient(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "TriggerClient",// the name by which the collector identifies the client
		  "localhost",// the name of the computer hosting the collector
		  9090        // the port at which the collector listens
		  ),QTFailed(0),QTCritical(0)
{
        url = getApplicationDescriptor()->getContextDescriptor()->getURL() + "/" + getApplicationDescriptor()->getURN();
	webInterface_p = new TriggerWebInterface(getContextURL(),getApplicationURL(), url ,  & mui_);
 
	xgi::bind(this, &L1TClient::handleWebRequest, "Request");


	xgi::bind(this, &L1TClient::CreateDQMPage, "DQMpage");
	xgi::bind(this, &L1TClient::CreateMenuPage, "menu");
	xgi::bind(this, &L1TClient::CreateStatusPage, "status");
	xgi::bind(this, &L1TClient::CreateDisplayPage, "display");

	logFile.open("L1TClient.log");
	
	logFile << "L1TClient::L1TClient: created pointer to TriggerWebInterface" << std::endl;		
	logFile << "L1TClient::L1TClient: url = " << url <<std::endl;		
	
	qtestsConfigured=false;
	meListConfigured=false;
	
	subscriber=new SubscriptionHandle;
	qtHandler=new QTestHandle;
	qtestalreadyrunning=false;
}



void L1TClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception){
	        webInterface_p->Default(in, out);
//		logFile << "L1TClient::general: Default() callback function" << std::endl;		
}




void L1TClient::handleWebRequest(xgi::Input * in, xgi::Output * out){
	webInterface_p->handleRequest(in, out);
	logFile <<"request: "<< webInterface_p->requestType() <<std::endl;
	if (webInterface_p->requestType() == "QTestConfigure")   {
		qtestsConfigured=false;
		if(!qtHandler->configureTests("QualityTests.xml",mui_)){
			qtestsConfigured=true;
//			qtHandler->attachTests(mui_);	
		}
	}
}
	
void L1TClient::CreateDQMPage(xgi::Input * in, xgi::Output * out){
	webInterface_p->CreateWI(in, out);
}
void L1TClient::CreateMenuPage(xgi::Input * in, xgi::Output * out){
	webInterface_p->CreateMenu(in, out);
}
void L1TClient::CreateStatusPage(xgi::Input * in, xgi::Output * out){
	webInterface_p->CreateStatus(in, out);
}
void L1TClient::CreateDisplayPage(xgi::Input * in, xgi::Output * out){
	webInterface_p->CreateDisplay(in, out);
}



void L1TClient::configure(){
	
	meListConfigured=false;
	qtestsConfigured=false;

	if(!subscriber->getMEList("MESubscriptionList.xml")) meListConfigured=true; 
	if(!qtHandler->configureTests("QualityTests.xml",mui_)) qtestsConfigured=true; 

}

void L1TClient::newRun(){
	upd_->registerObserver(this);
	

}

void L1TClient::endRun(){

}

void L1TClient::onUpdate() const{
	// put here the code that needs to be executed on every update:
	std::vector<std::string> uplist;
	mui_->getUpdatedContents(uplist);
	
	if(meListConfigured) subscriber->updateSubscriptions(mui_);
	if(qtestsConfigured){	
	
        if(!qtestalreadyrunning){
	///ME's subscription
	    if(meListConfigured) {
		  subscriber->makeSubscriptions(mui_);
	    }else{
	    }
	    ///QT's enabling
	    if(qtestsConfigured){
	    	    qtHandler->attachTests(mui_);
	    }else{
	    }
//			qtHandler->attachTests(mui_);
			qtestalreadyrunning=true;
            }			
		
		if(webInterface_p->globalQTStatusRequest()) qtHandler->checkGolbalQTStatus(mui_);
		
		if(webInterface_p->detailedQTStatusRequest()) {
			 
			 std::map< std::string, std::vector<std::string> > theAlarms=qtHandler->checkDetailedQTStatus(mui_);
			 
			 for(std::map<std::string,std::vector<std::string> >::iterator itr=theAlarms.begin();itr!=theAlarms.end();++itr){
			 	std::string alarmType=	itr->first;
				std::vector<std::string> messages=itr->second;
				for(std::vector<std::string>::iterator message=messages.begin();message!=messages.end();++message ){
					logFile <<*message<<std::endl;
				}
			 }
		}
		
	}
}











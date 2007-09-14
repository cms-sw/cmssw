/** \file
 *
 *  Implementation of  MuonDQMClient
 *
 *  $Date: 2006/07/20 16:17:46 $
 *  $Revision: 1.5 $
 *  \author Ilaria Segoni
 */



#include "DQM/RPCMonitorClient/interface/MuonDQMClient.h"
#include "DQM/RPCMonitorClient/interface/DQMClientDefineDebug.h"

#include "DQMServices/ClientConfig/interface/SubscriptionHandle.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"


MuonDQMClient::MuonDQMClient(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "MuonClient",// the name by which the collector identifies the client
		  "localhost",// the name of the computer hosting the collector
		  9090        // the port at which the collector listens
		  ),QTFailed(0),QTCritical(0)
{
	webInterface_p = new MuonWebInterface(getContextURL(),getApplicationURL(), & mui_);
  
	xgi::bind(this, &MuonDQMClient::handleWebRequest, "Request");

	logFile.open("MuonDQMClient.log");
	qtestsConfigured=false;
	meListConfigured=false;
	
	subscriber=new SubscriptionHandle;
	qtHandler=new QTestHandle;
}



void MuonDQMClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception){
	webInterface_p->Default(in, out);
}




void MuonDQMClient::handleWebRequest(xgi::Input * in, xgi::Output * out){
	webInterface_p->handleRequest(in, out);
	logFile <<"request: "<<webInterface_p->requestType() <<std::endl;
	if (webInterface_p->requestType() == "QTestConfigure")   {
		qtestsConfigured=false;
		if(!qtHandler->configureTests("QualityTests.xml",mui_->getBEInterface())){
			qtestsConfigured=true;
			qtHandler->attachTests(mui_->getBEInterface());			
		}
	}		
}



void MuonDQMClient::configure(){
	
	meListConfigured=false;
	qtestsConfigured=false;

	if(!subscriber->getMEList("MESubscriptionList.xml")) meListConfigured=true; 
	if(!qtHandler->configureTests("QualityTests.xml",mui_->getBEInterface())) qtestsConfigured=true; 
}

void MuonDQMClient::newRun(){
	upd_->registerObserver(this);
	
	///ME's subscription
	if(meListConfigured) {
		subscriber->makeSubscriptions(mui_);
	}else{
		logFile << "Cannot subscribe to ME's, error occurred in configuration." << std::endl;		
	}
	///QT's enabling
	if(qtestsConfigured){
		qtHandler->attachTests(mui_->getBEInterface());	
	}else{
		logFile << "Cannot run quality tests, error occurred in configuration." << std::endl;		
	}

}

void MuonDQMClient::endRun(){

}

void MuonDQMClient::onUpdate() const{
	// put here the code that needs to be executed on every update:
	std::vector<std::string> uplist;
	mui_->getBEInterface()->getUpdatedContents(uplist);
	
	if(meListConfigured) subscriber->updateSubscriptions(mui_);
	if(qtestsConfigured){	
		
		if(webInterface_p->globalQTStatusRequest()) 
		  qtHandler->checkGlobalQTStatus(mui_->getBEInterface());
		if(webInterface_p->detailedQTStatusRequest()) {
			 std::map< std::string, std::vector<std::string> > theAlarms=qtHandler->checkDetailedQTStatus(mui_->getBEInterface());
			 for(std::map<std::string,std::vector<std::string> >::iterator itr=theAlarms.begin();itr!=theAlarms.end();++itr){
			 	std::string alarmType=	itr->first;
				std::vector<std::string> messages=itr->second;
				logFile <<"Error Type: "<<alarmType<<std::endl;
				for(std::vector<std::string>::iterator message=messages.begin();message!=messages.end();++message ){
					logFile <<*message<<std::endl;
				}
			 }
		}
		
	}
}










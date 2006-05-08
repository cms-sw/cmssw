/** \file
 *
 *  Implementation of  MuonDQMClient
 *
 *  $Date: 2006/05/04 10:27:25 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */



#include "DQM/RPCMonitorClient/interface/MuonDQMClient.h"
#include "DQM/RPCMonitorClient/interface/DQMClientDefineDebug.h"

#include "DQM/RPCMonitorClient/interface/SubscriptionHandle.h"
#include "DQM/RPCMonitorClient/interface/QTestHandle.h"


MuonDQMClient::MuonDQMClient(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "test",     // the name by which the collector identifies the client
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
}



void MuonDQMClient::configure(){
	
	meListConfigured=false;
	qtestsConfigured=false;

	#ifdef RPC_CLIENT_DEBUG
	logFile << "Configuring MuonDQMClient" << std::endl;
	#endif
	
	if(!subscriber->configure("MESubscriptionList.xml")) meListConfigured=true; 
	if(!qtHandler->configure("QualityTests.xml",mui_)) qtestsConfigured=true; 
}

void MuonDQMClient::newRun(){
	upd_->registerObserver(this);
	
	///ME's subscription
	if(meListConfigured) {
		subscriber->enable(mui_);
	}else{
		logFile << "Cannot subscribe to ME's, error occurred in configuration." << std::endl;		
	}
	///QT's enabling
	if(qtestsConfigured){
		qtHandler->enable(mui_);	
	}else{
		logFile << "Cannot run quality tests, error occurred in configuration." << std::endl;		
	}


}

void MuonDQMClient::endRun(){

}

void MuonDQMClient::onUpdate() const{
	// put here the code that needs to be executed on every update:
	std::vector<std::string> uplist;
	mui_->getUpdatedContents(uplist);
	
	if(meListConfigured) subscriber->onUpdate(mui_);
	if(qtestsConfigured){	
		QTestHandle::onUpdateAction action;
		
		if(webInterface_p->globalQTStatusRequest())   {
			action=QTestHandle::checkGlobal;
			qtHandler->onUpdate(action,mui_);
		}	
		
		if(webInterface_p->detailedQTStatusRequest()) {
			action=QTestHandle::checkDetailed;
			qtHandler->onUpdate(action,mui_);
		}
	}
}










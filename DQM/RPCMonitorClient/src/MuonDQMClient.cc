/** \file
 *
 *  Implementation of  MuonDQMClient
 *
 *  $Date: 2006/04/24 10:00:17 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */



#include "DQM/RPCMonitorClient/interface/MuonDQMClient.h"
#include "DQM/RPCMonitorClient/interface/DQMClientDefineDebug.h"

#include "DQM/RPCMonitorClient/interface/QTestConfigurationParser.h"
#include "DQM/RPCMonitorClient/interface/QTestConfigure.h"
#include "DQM/RPCMonitorClient/interface/QTestEnabler.h"
#include "DQM/RPCMonitorClient/interface/QTestStatusChecker.h"
#include "DQM/RPCMonitorClient/interface/MESubscriptionParser.h"
#include "DQM/RPCMonitorClient/interface/SubscriptionHandle.h"

#include "DQM/RPCMonitorClient/interface/DQMParserBase.h"

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
		
	qtParser     = new QTestConfigurationParser();
	meListParser = new  MESubscriptionParser();

	qtConfigurer = new QTestConfigure();
	qtEnabler    = new QTestEnabler();
	qtChecker    = new QTestStatusChecker();

}

void MuonDQMClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception){
	webInterface_p->Default(in, out);
}


void MuonDQMClient::handleWebRequest(xgi::Input * in, xgi::Output * out){
	webInterface_p->handleRequest(in, out);
}

void MuonDQMClient::configure(){
	
	qtestsConfigured=false;
	meListConfigured=false;

	#ifdef RPC_CLIENT_DEBUG
	logFile << "Configuring MuonDQMClient" << std::endl;
	#endif
	
	///Configuring ME subscription list
	std::string  meListFile="MESubscriptionList.xml";
	meListParser->getDocument(meListFile);
	if(! meListParser->parseMESubscription() ) meListConfigured=true;
	
	///Configuring Quality Tests
	std::string xmlFile="QualityTests.xml";  
	qtParser->getDocument(xmlFile);
	if(! qtParser->parseQTestsConfiguration() ){
	      std::map<std::string, std::map<std::string, std::string> > testsList=qtParser->testsList();
	      if(! qtConfigurer->enableTests(testsList,mui_)){
			qtestsConfigured=true;
			
			#ifdef RPC_CLIENT_DEBUG
		    	std::vector<std::string> tests= qtConfigurer->testsReady();
		    	for(std::vector<std::string>::iterator itr=tests.begin();  itr!=tests.end();++itr){
				logFile<<"Configured Test: "<<*itr<<std::endl;
		    	}	      
			#endif
	      
	      }else{
		    	logFile<< "In MuonDQMClient::configure, Error Configuring Quality Tests"<<std::endl;
		    	return;
	      }
	}else{
	      logFile<< "In MuonDQMClient::configure, Error Parsing Quality Tests"<<std::endl;
	      return;
	}

}

void MuonDQMClient::newRun(){
	upd_->registerObserver(this);
	if(meListConfigured){
		std::vector<std::string> subList=meListParser->subscribeList();
		std::vector<std::string> unsubList=meListParser->unsubscribeList();
		SubscriptionHandle subscriber;
		subscriber.setSubscriptions(subList, unsubList, mui_);	
	}
	if(qtestsConfigured){	
		#ifdef RPC_CLIENT_DEBUG
		logFile << "Beginning to run quality tests." << std::endl;
		#endif
	
		std::map<std::string, std::vector<std::string> > mapMeToTests= qtParser->meToTestsList();
		qtEnabler->startTests(mapMeToTests, mui_);
	}else{
		logFile << "Cannot run quality tests, error occurred in configuration." << std::endl;
		
	}
}

void MuonDQMClient::endRun(){
//	delete qtParser;
//	delete qtConfigurer;
//	delete qtEnabler;
//	delete qtChecker;
//	qtParser=0;
//	qtConfigurer=0;
//	qtEnabler=0;
//	qtChecker=0;

}

void MuonDQMClient::onUpdate() const{
	// put here the code that needs to be executed on every update:
	std::vector<std::string> uplist;
	mui_->getUpdatedContents(uplist);
	
	if(webInterface_p->globalQTStatusRequest())   this->checkGolbalQTStatus();
	
	if(webInterface_p->detailedQTStatusRequest()) this->checkDetailedQTStatus();
}



void MuonDQMClient::checkGolbalQTStatus() const{

	std::pair<std::string,std::string> globalStatus = qtChecker->checkGlobalStatus(mui_);
	//WebMessage * message= new WebMessage(getApplicationURL(), "350px" , "500px", globalStatus.first,globalStatus.second  );
	logFile <<"Quality Tests global status: " <<globalStatus.first<<std::endl;

}


void MuonDQMClient::checkDetailedQTStatus() const {

		std::map< std::string, std::vector<std::string> > messages= qtChecker->checkDetailedStatus(mui_);  
		///Error messages  
		char alarm[128] ;
		std::vector<std::string> errors = messages["red"];
		QTFailed=errors.size();
		sprintf(alarm,"Number of Errors :%d",QTFailed);
		logFile<< alarm <<std::endl;
		
		//WebMessage * messageNumberOfErrors= new WebMessage(getApplicationURL(), "380px" , "500px", alarm,"red"  );
		//page->add("numberoferrors",messageNumberOfErrors);
		///Warning messages  
		std::vector<std::string> warnings = messages["orange"];
		QTCritical=warnings.size();
		sprintf(alarm,"Number of Warnings :%d", QTCritical);
		logFile<< alarm <<std::endl;
		//WebMessage * messageNumberOfWarningss= new WebMessage(getApplicationURL(), "410px" , "500px", alarm,"orange"  );
		//page->add("numberofwarnings",messageNumberOfWarningss );
 

}







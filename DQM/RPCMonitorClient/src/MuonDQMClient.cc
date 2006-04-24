/** \file
 *
 *  Implementation of  MuonDQMClient
 *
 *  $Date: 2006/04/05 15:45:24 $
 *  $Revision: 1.5 $
 *  \author Ilaria Segoni
 */



#include "DQM/RPCMonitorClient/interface/MuonDQMClient.h"
#include "DQM/RPCMonitorClient/interface/QTestConfigurationParser.h"
#include "DQM/RPCMonitorClient/interface/QTestConfigure.h"
#include "DQM/RPCMonitorClient/interface/QTestEnabler.h"
#include "DQM/RPCMonitorClient/interface/QTestStatusChecker.h"
#include "DQM/RPCMonitorClient/interface/DQMClientDefineDebug.h"


MuonDQMClient::MuonDQMClient(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "test",     // the name by which the collector identifies the client
		  "localhost",// the name of the computer hosting the collector
		  9090        // the port at which the collector listens
		  )
{
	webInterface_p = new MuonWebInterface(getContextURL(),getApplicationURL(), & mui_);
  
	xgi::bind(this, &MuonDQMClient::handleWebRequest, "Request");

}

void MuonDQMClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
	webInterface_p->Default(in, out);
}


void MuonDQMClient::handleWebRequest(xgi::Input * in, xgi::Output * out)
{
	webInterface_p->handleRequest(in, out);
}

void MuonDQMClient::configure()
{
	qtParser     = new QTestConfigurationParser();
	qtConfigurer = new QTestConfigure();
	qtEnabler    = new QTestEnabler();
	qtChecker    = new QTestStatusChecker();

	#ifdef DEBUG
	std::cout << "Quality Tests are being configured" << std::endl;
	#endif
	std::string xmlFile="QualityTests.xml";  
	if(! qtParser->parseQTestsConfiguration(xmlFile) ){
	      std::map<std::string, std::map<std::string, std::string> > testsList=qtParser->testsList();
	      if(! qtConfigurer->enableTests(testsList,mui_)){
		    std::vector<std::string> tests= qtConfigurer->testsReady();
		    for(std::vector<std::string>::iterator itr=tests.begin();  itr!=tests.end();++itr){
			    std::cout<<"Configured Test: "<<*itr<<std::endl;
		    }	      
	      }else{
		    std::cout<< "Error Configuring Quality Tests"<<std::endl;
		    return;
	      }
	}else{
	      std::cout<< "Error Parsing Quality Tests"<<std::endl;
	      return;
	}

}

void MuonDQMClient::newRun()
{
	upd_->registerObserver(this);
	#ifdef DEBUG
	std::cout << "Beginning to run quality tests" << std::endl;
	#endif
	std::map<std::string, std::vector<std::string> > mapMeToTests= qtParser->meToTestsList();
	qtEnabler->startTests(mapMeToTests, mui_);
}

void MuonDQMClient::endRun()
{
//	delete qtParser;
//	delete qtConfigurer;
//	delete qtEnabler;
//	delete qtChecker;
//	qtParser=0;
//	qtConfigurer=0;
//	qtEnabler=0;
//	qtChecker=0;

}

void MuonDQMClient::onUpdate() const
{
	// put here the code that needs to be executed on every update:
	std::vector<std::string> uplist;
	mui_->getUpdatedContents(uplist);
	
	if(webInterface_p->globalQTStatusRequest()){
 		std::pair<std::string,std::string> globalStatus = qtChecker->checkGlobalStatus(mui_);
   		//WebMessage * message= new WebMessage(getApplicationURL(), "350px" , "500px", globalStatus.first,globalStatus.second  );
   		#ifdef DEBUG
   		std::cout<<"Quality Tests global status: " <<globalStatus.first<<std::endl;
   		#endif
	}
	
	if(webInterface_p->detailedQTStatusRequest()){
  
		std::map< std::string, std::vector<std::string> > messages= qtChecker->checkDetailedStatus(mui_);  
		///Error messages  
		char alarm[128] ;
		std::vector<std::string> errors = messages["red"];
		sprintf(alarm,"Number of Errors :%d",errors.size());
		#ifdef DEBUG
		std::cout<< alarm <<std::endl;
		#endif
		//WebMessage * messageNumberOfErrors= new WebMessage(getApplicationURL(), "380px" , "500px", alarm,"red"  );
		//page->add("numberoferrors",messageNumberOfErrors);
		///Warning messages  
		std::vector<std::string> warnings = messages["orange"];
		sprintf(alarm,"Number of Warnings :%d", warnings.size() );
		#ifdef DEBUG
		std::cout<< alarm <<std::endl;
		#endif
		//WebMessage * messageNumberOfWarningss= new WebMessage(getApplicationURL(), "410px" , "500px", alarm,"orange"  );
		//page->add("numberofwarnings",messageNumberOfWarningss );
 
	}
}













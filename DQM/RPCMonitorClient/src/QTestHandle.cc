/** \file
 *
 *  Implementation of  QTestHandle
 *
 *  $Date: 2006/05/04 10:27:25 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */


#include "DQM/RPCMonitorClient/interface/QTestHandle.h"
#include "DQM/RPCMonitorClient/interface/QTestConfigurationParser.h"
#include "DQM/RPCMonitorClient/interface/QTestConfigure.h"
#include "DQM/RPCMonitorClient/interface/QTestEnabler.h"
#include "DQM/RPCMonitorClient/interface/QTestStatusChecker.h"
#include "DQM/RPCMonitorClient/interface/DQMClientDefineDebug.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

QTestHandle::QTestHandle(){
	qtParser     = new QTestConfigurationParser();
	qtConfigurer = new QTestConfigure();
	qtEnabler    = new QTestEnabler();
	qtChecker    = new QTestStatusChecker();
}

QTestHandle::~QTestHandle(){
	delete qtParser;     
	delete qtConfigurer; 
	delete qtEnabler;   
	delete qtChecker;    
}

bool QTestHandle::configure(std::string configFile, MonitorUserInterface * mui){
	
	qtParser->getDocument(configFile);
	if(! qtParser->parseQTestsConfiguration() ){
	      std::map<std::string, std::map<std::string, std::string> > testsList=qtParser->testsList();
	      if(! qtConfigurer->enableTests(testsList,mui)){
			
			#ifdef RPC_CLIENT_DEBUG
		    	std::vector<std::string> tests= qtConfigurer->testsReady();
		    	for(std::vector<std::string>::iterator itr=tests.begin();  itr!=tests.end();++itr){
				//logFile<<"Configured "<<*itr<<" test"<<std::endl;
				std::cout<<"Configured "<<*itr<<" test"<<std::endl;
		    	}	      
			#endif
	      
	      }else{
		    	//logFile<< "In MuonDQMClient::configure, Error Configuring Quality Tests"<<std::endl;
		    	std::cout<< "In MuonDQMClient::configure, Error Configuring Quality Tests"<<std::endl;
		    	return true;
	      }
	}else{
	      //logFile<< "In MuonDQMClient::configure, Error Parsing Quality Tests"<<std::endl;
	      std::cout<< "In MuonDQMClient::configure, Error Parsing Quality Tests"<<std::endl;
	      return true;
	}
	
	return false;


}

void QTestHandle::enable(MonitorUserInterface * mui){
		std::map<std::string, std::vector<std::string> > mapMeToTests= qtParser->meToTestsList();
		qtEnabler->startTests(mapMeToTests, mui);
}

void QTestHandle::onUpdate(QTestHandle::onUpdateAction action,MonitorUserInterface * mui){
	if(action == QTestHandle::checkGlobal) this->checkGolbalQTStatus(mui);
	if(action == QTestHandle::checkDetailed) this->checkDetailedQTStatus(mui);
}




void QTestHandle::checkGolbalQTStatus(MonitorUserInterface * mui) const{

	std::pair<std::string,std::string> globalStatus = qtChecker->checkGlobalStatus(mui);
	//WebMessage * message= new WebMessage(getApplicationURL(), "350px" , "500px", globalStatus.first,globalStatus.second  );
	//logFile <<"Quality Tests global status: " <<globalStatus.first<<std::endl;
	std::cout <<"Quality Tests global status: " <<globalStatus.first<<std::endl;

}


void QTestHandle::checkDetailedQTStatus(MonitorUserInterface * mui) const {

		std::map< std::string, std::vector<std::string> > messages= qtChecker->checkDetailedStatus(mui);  
		///Error messages  
		char alarm[128] ;
		std::vector<std::string> errors = messages["red"];
		sprintf(alarm,"Number of Errors :%d",errors.size());
		//logFile<< alarm <<std::endl;
		std::cout<< alarm <<std::endl;
		
		//WebMessage * messageNumberOfErrors= new WebMessage(getApplicationURL(), "380px" , "500px", alarm,"red"  );
		//page->add("numberoferrors",messageNumberOfErrors);
		///Warning messages  
		std::vector<std::string> warnings = messages["orange"];
		sprintf(alarm,"Number of Warnings :%d", warnings.size());
		//logFile<< alarm <<std::endl;
		std::cout<< alarm <<std::endl;
		//WebMessage * messageNumberOfWarningss= new WebMessage(getApplicationURL(), "410px" , "500px", alarm,"orange"  );
		//page->add("numberofwarnings",messageNumberOfWarningss );
 

}

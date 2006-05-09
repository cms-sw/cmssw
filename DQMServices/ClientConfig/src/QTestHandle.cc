/** \file
 *
 *  Implementation of  QTestHandle
 *
 *  $Date: 2006/05/04 10:27:25 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */


#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include "DQMServices/ClientConfig/interface/QTestConfigurationParser.h"
#include "DQMServices/ClientConfig/interface/QTestConfigure.h"
#include "DQMServices/ClientConfig/interface/QTestStatusChecker.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

QTestHandle::QTestHandle(){
	qtParser     = new QTestConfigurationParser();
	qtConfigurer = new QTestConfigure();
	qtChecker    = new QTestStatusChecker();
}

QTestHandle::~QTestHandle(){
	delete qtParser;     
	delete qtConfigurer; 
	delete qtChecker; 
	   
}

bool QTestHandle::configureTests(std::string configFile, MonitorUserInterface * mui){
	
	qtParser->getDocument(configFile);
	if(! qtParser->parseQTestsConfiguration() ){
	      std::map<std::string, std::map<std::string, std::string> > testsList=qtParser->testsList();
	      if(qtConfigurer->enableTests(testsList,mui)) return true;
	
	}else{
	      return true;
	}
	
	return false;


}

void QTestHandle::attachTests(MonitorUserInterface * mui){
		std::map<std::string, std::vector<std::string> > mapMeToTests= qtParser->meToTestsList();

	for(std::map<std::string, std::vector<std::string> >::iterator itr = mapMeToTests.begin();
	          itr != mapMeToTests.end();++itr){   
	    
		std::string meName=itr->first;
		mui->subscribe(meName);
		std::vector<std::string> tests=itr->second;
		for(std::vector<std::string>::iterator testsItr=tests.begin(); 
			testsItr!=tests.end(); ++testsItr){
			mui->useQTest(meName, *testsItr);
		}	
	}


}

void QTestHandle::checkGolbalQTStatus(MonitorUserInterface * mui) const{

	std::pair<std::string,std::string> globalStatus = qtChecker->checkGlobalStatus(mui);
	std::cout <<"Quality Tests global status: " <<globalStatus.first<<std::endl;

}


void QTestHandle::checkDetailedQTStatus(MonitorUserInterface * mui) const {

		std::map< std::string, std::vector<std::string> > messages= qtChecker->checkDetailedStatus(mui);  

		char alarm[128] ;
		std::vector<std::string> errors = messages["red"];
		sprintf(alarm,"Number of Errors :%d",errors.size());
		std::cout<< alarm <<std::endl;
		
		std::vector<std::string> warnings = messages["orange"];
		sprintf(alarm,"Number of Warnings :%d", warnings.size());
		std::cout<< alarm <<std::endl;
}

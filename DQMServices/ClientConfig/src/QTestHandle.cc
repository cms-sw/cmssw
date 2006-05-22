/** \file
 *
 *  Implementation of  QTestHandle
 *
 *  $Date: 2006/05/09 21:28:37 $
 *  $Revision: 1.1 $
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

std::pair<std::string,std::string> QTestHandle::checkGolbalQTStatus(MonitorUserInterface * mui) const{

	return qtChecker->checkGlobalStatus(mui);

}


std::map< std::string, std::vector<std::string> > QTestHandle::checkDetailedQTStatus(MonitorUserInterface * mui) const {

		return qtChecker->checkDetailedStatus(mui);
		
}

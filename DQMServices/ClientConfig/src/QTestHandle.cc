/** \file
 *
 *  Implementation of  QTestHandle
 *
 *  $Date: 2007/07/08 21:03:54 $
 *  $Revision: 1.3.4.1 $
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
	
	testsConfigured = false;
}

QTestHandle::~QTestHandle(){
	delete qtParser;     
	delete qtConfigurer; 
	delete qtChecker; 
	   
}

bool QTestHandle::configureTests(std::string configFile, DaqMonitorBEInterface *
bei){
	
	if(testsConfigured) {
		qtParser->getNewDocument(configFile);
	}else{	
		qtParser->getDocument(configFile);
		testsConfigured=true;
	}

	if(! qtParser->parseQTestsConfiguration() ){
	      std::map<std::string, std::map<std::string, std::string> > testsONList=qtParser->testsList();
	      std::vector<std::string> testsOFFList=qtParser->testsOff();
	      qtConfigurer->desableTests(testsOFFList,bei);
	      if(qtConfigurer->enableTests(testsONList,bei)) return true;
	
	}else{
	      return true;
	}
	
	return false;


}

void QTestHandle::attachTests(DaqMonitorBEInterface * bei){
		std::map<std::string, std::vector<std::string> > mapMeToTests= qtParser->meToTestsList();

	for(std::map<std::string, std::vector<std::string> >::iterator itr = mapMeToTests.begin();
	          itr != mapMeToTests.end();++itr){   
	    
		std::string meName=itr->first;
//		bei->subscribe(meName);
		std::vector<std::string> tests=itr->second;
		for(std::vector<std::string>::iterator testsItr=tests.begin(); 
			testsItr!=tests.end(); ++testsItr){
			bei->useQTest(meName, *testsItr);
		}	
	}


}

std::pair<std::string,std::string>
QTestHandle::checkGlobalQTStatus(DaqMonitorBEInterface * bei) const{

	return qtChecker->checkGlobalStatus(bei);

}


std::map< std::string, std::vector<std::string> >
QTestHandle::checkDetailedQTStatus(DaqMonitorBEInterface * bei) const {

		return qtChecker->checkDetailedStatus(bei);
		
}

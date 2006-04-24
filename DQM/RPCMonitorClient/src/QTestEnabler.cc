/** \file
 *
 *  Implementation of QTestEnabler
 *
 *  $Date: 2006/04/05 15:45:33 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */
#include "DQM/RPCMonitorClient/interface/QTestEnabler.h"

void QTestEnabler::startTests(std::map<std::string, std::vector<std::string> > mapMeToTests, MonitorUserInterface * mui){

	logFile<<"\nIn QTestEnabler::startTests"<<std::endl;
	for(std::map<std::string, std::vector<std::string> >::iterator itr = mapMeToTests.begin();
	          itr != mapMeToTests.end();++itr){   
	    
		std::string meName=itr->first;
		logFile<<"subscribing to "<<meName<<std::endl;
		mui->subscribe(meName);
		logFile<<"done subscription "<<meName<<std::endl;
		std::vector<std::string> tests=itr->second;
		for(std::vector<std::string>::iterator testsItr=tests.begin(); 
			testsItr!=tests.end(); ++testsItr){
			logFile<<"Quality tests attached: "<<*testsItr <<std::endl;
			mui->useQTest(meName, *testsItr);
		}		
	}
}


/** \file
 *
 *  Implementation of QTestEnabler
 *
 *  $Date: 2006/04/24 10:04:08 $
 *  $Revision: 1.3 $
 *  \author Ilaria Segoni
 */
#include "DQM/RPCMonitorClient/interface/QTestEnabler.h"
#include "DQM/RPCMonitorClient/interface/QTestDefineDebug.h"

void QTestEnabler::startTests(std::map<std::string, std::vector<std::string> > mapMeToTests, MonitorUserInterface * mui){

	for(std::map<std::string, std::vector<std::string> >::iterator itr = mapMeToTests.begin();
	          itr != mapMeToTests.end();++itr){   
	    
		std::string meName=itr->first;
		mui->subscribe(meName);
		#ifdef QT_MANAGING_DEBUG	
		std::cout<<"Quality tests for ME "<<meName<<": "<<std::endl;
		#endif
		std::vector<std::string> tests=itr->second;
		for(std::vector<std::string>::iterator testsItr=tests.begin(); 
			testsItr!=tests.end(); ++testsItr){
			#ifdef QT_MANAGING_DEBUG	
			std::cout<<*testsItr <<", ";
			#endif
			mui->useQTest(meName, *testsItr);
		}	
		#ifdef QT_MANAGING_DEBUG	
		std::cout<<std::endl;
		#endif
	}
}


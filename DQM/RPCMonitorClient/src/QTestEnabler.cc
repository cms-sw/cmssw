/** \file
 *
 *  Implementation of QTestEnabler
 *
 *  $Date: 2006/03/14 11:24:20 $
 *  $Revision: 1.6 $
 *  \author Ilaria Segoni
 */
#include "DQM/RPCMonitorClient/interface/QTestEnabler.h"
#include "DQM/RPCMonitorClient/interface/DQMQualityTestsConfiguration.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"


void QTestEnabler::enableTests(std::map<std::string, std::map<std::string, std::string> > tests,MonitorUserInterface * mui){
	
	testsEnabled.clear();
	
	std::cout<<"\nENABLING TESTS"<<std::endl;
	std::map<std::string, std::map<std::string, std::string> >::iterator itr;
	for(itr= tests.begin(); itr!= tests.end();++itr){
 
		std::cout<<"\nTEST NAME="<<itr->first<<std::endl;
		std::map<std::string, std::string> params= itr->second;
  
		std::string testName = itr->first; 
		std::string testType = params[dqm::qtest_config::type]; 
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::XRangeContent.c_str())) this->EnableXRangeTest(testName, params,mui);       
	}
}



void QTestEnabler::EnableXRangeTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){

	std::cout<<"\nIn QTestEnabler::EnableXRangeTest, test Name= "<<testName <<std::endl;
	testsEnabled.push_back(testName);

	QCriterion * qc1 = mui->createQTest(ContentsXRangeROOT::getAlgoName(),testName);
	MEContentsXRangeROOT * me_qc1 = (MEContentsXRangeROOT *) qc1;
	double xmin=atof(params["xmin"].c_str());
	double xmax=atof(params["xmax"].c_str());
	double warning=atof(params["warning"].c_str());
	me_qc1->setAllowedXRange(xmin,xmax);
	me_qc1->setWarningProb(warning);
}


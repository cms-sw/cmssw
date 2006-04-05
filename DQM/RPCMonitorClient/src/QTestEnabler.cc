/** \file
 *
 *  Implementation of QTestEnabler
 *
 *  $Date: 2006/04/05 08:05:37 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */
#include "DQM/RPCMonitorClient/interface/QTestEnabler.h"
#include "DQM/RPCMonitorClient/interface/DQMQualityTestsConfiguration.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

bool QTestEnabler::enableTests(std::map<std::string, std::map<std::string, std::string> > tests,MonitorUserInterface * mui){
	
	testsEnabled.clear();
	
	std::cout<<"\nENABLING TESTS"<<std::endl;
	std::map<std::string, std::map<std::string, std::string> >::iterator itr;
	for(itr= tests.begin(); itr!= tests.end();++itr){
 
		std::cout<<"\nTEST NAME="<<itr->first<<std::endl;
		std::map<std::string, std::string> params= itr->second;
  
		std::string testName = itr->first; 
		std::string testType = params[dqm::qtest_config::type]; 
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::XRangeContent.c_str())) this->EnableXRangeTest(testName, params,mui);       
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::YRangeContent.c_str())) this->EnableYRangeTest(testName, params,mui);       
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::DeadChannel.c_str()))   this->EnableDeadChannelTest(testName, params,mui);       
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::NoisyChannel.c_str()))  this->EnableNoisyChannelTest(testName, params,mui);       
	}
	
	return false;	
}


void QTestEnabler::startTests(std::map<std::string, std::vector<std::string> > mapMeToTests, MonitorUserInterface * mui){

	std::cout<<"\nIn QTestEnabler::startTests"<<std::endl;
	for(std::map<std::string, std::vector<std::string> >::iterator itr = mapMeToTests.begin();
	          itr != mapMeToTests.end();++itr){   
	    
		std::string meName=itr->first;
		mui->subscribe( meName);
		std::vector<std::string> tests=itr->second;
		for(std::vector<std::string>::iterator testsItr=tests.begin(); 
			testsItr!=tests.end(); ++testsItr){
			mui->useQTest(meName, *testsItr);
		}		
	}
}

void QTestEnabler::EnableXRangeTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){
	std::cout<<"In QTestEnabler::EnableXRangeTest, test Name= "<<testName <<std::endl;
	testsEnabled.push_back(testName);

	QCriterion * qc1 = mui->createQTest(ContentsXRangeROOT::getAlgoName(),testName);
	MEContentsXRangeROOT * me_qc1 = (MEContentsXRangeROOT *) qc1;
	double xmin=atof(params["xmin"].c_str());
	double xmax=atof(params["xmax"].c_str());
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setAllowedXRange(xmin,xmax);
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestEnabler::EnableYRangeTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){
	std::cout<<"In In QTestEnabler::EnableYRangeTest, test Name= "<<testName <<std::endl;
	testsEnabled.push_back(testName);
	std::cout<<"sonoqui00 "<< ContentsYRangeROOT::getAlgoName()<<std::endl;
	QCriterion * qc1 = mui->createQTest(ContentsYRangeROOT::getAlgoName(),testName);
	MEContentsYRangeROOT * me_qc1 = (MEContentsYRangeROOT *) qc1;
	double ymin=atof(params["ymin"].c_str());
	double ymax=atof(params["ymax"].c_str());
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setAllowedYRange(ymin,ymax);
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestEnabler::EnableDeadChannelTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){
	std::cout<<"In QTestEnabler::EnableDeadChannelTest, test Name= "<<testName <<std::endl;
	testsEnabled.push_back(testName);
	std::cout<<"sonoqui0 "<< DeadChannelROOT::getAlgoName() <<std::endl;

	QCriterion * qc1 = mui->createQTest(DeadChannelROOT::getAlgoName(),testName);
	std::cout<<"sonoqui1 " <<std::endl;
	MEContentsXRangeROOT * me_qc1 = ( MEContentsXRangeROOT *) qc1;
	std::cout<<"sonoqui2 " <<std::endl;
	unsigned int threshold=(unsigned int)atof(params["threshold"].c_str());
	std::cout<<"sonoqui3 " <<threshold<<std::endl;
	me_qc1->setMinimumEntries(7);
	//double warning=atof(params["warning"].c_str());
	//double error=atof(params["error"].c_str());
	//me_qc1->setWarningProb(warning);
	//me_qc1->setErrorProb(error);
}

void QTestEnabler::EnableNoisyChannelTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){
	std::cout<<"In QTestEnabler::EnableNoisyChannelTest, test Name= "<<testName <<std::endl;
	testsEnabled.push_back(testName);

	QCriterion * qc1 = mui->createQTest(NoisyChannelROOT::getAlgoName(),testName);
	MENoisyChannelROOT * me_qc1 = ( MENoisyChannelROOT *) qc1;
	unsigned int neighbors=(unsigned int)atof(params["neighbours"].c_str());
	double tolerance=atof(params["tolerance"].c_str());
	me_qc1->setNumNeighbors (neighbors);
  	me_qc1->setTolerance (tolerance);
	
	//double warning=atof(params["warning"].c_str());
	//double error=atof(params["error"].c_str());
	//me_qc1->setWarningProb(warning);
	//me_qc1->setErrorProb(error);
}

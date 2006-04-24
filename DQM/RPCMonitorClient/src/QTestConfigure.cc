/** \file
 *
 *  Implementation of QTestConfigure
 *
 *  $Date: 2006/04/05 15:45:33 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */
#include "DQM/RPCMonitorClient/interface/QTestConfigure.h"
#include "DQM/RPCMonitorClient/interface/DQMQualityTestsConfiguration.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

bool QTestConfigure::enableTests(std::map<std::string, std::map<std::string, std::string> > tests,MonitorUserInterface * mui){
	logFile.open("QTConfigurer.log");
	
	testsConfigured.clear();
	
	logFile<<"\nENABLING TESTS"<<std::endl;
	std::map<std::string, std::map<std::string, std::string> >::iterator itr;
	for(itr= tests.begin(); itr!= tests.end();++itr){
 
		logFile<<"\nTEST NAME="<<itr->first<<std::endl;
		std::map<std::string, std::string> params= itr->second;
  
		std::string testName = itr->first; 
		std::string testType = params[dqm::qtest_config::type]; 
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::XRangeContent.c_str())) this->EnableXRangeTest(testName, params,mui);       
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::YRangeContent.c_str())) this->EnableYRangeTest(testName, params,mui);       
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::DeadChannel.c_str()))   this->EnableDeadChannelTest(testName, params,mui);       
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::NoisyChannel.c_str()))  this->EnableNoisyChannelTest(testName, params,mui);       
		if(!std::strcmp(testType.c_str(),dqm::qtest_config::MeanInExpectedValue.c_str()))  this->EnableMeanWithinExpectedTest(testName, params,mui);       
	}
	
	return false;	
}



void QTestConfigure::EnableXRangeTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){
	logFile<<"In QTestConfigure::EnableXRangeTest, test Name= "<<testName <<std::endl;
	testsConfigured.push_back(testName);

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

void QTestConfigure::EnableYRangeTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){
	logFile<<"In In QTestConfigure::EnableYRangeTest, test Name= "<<testName <<std::endl;
	testsConfigured.push_back(testName);
	logFile<<"sonoqui00 "<< ContentsYRangeROOT::getAlgoName()<<std::endl;
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

void QTestConfigure::EnableDeadChannelTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){
	logFile<<"In QTestConfigure::EnableDeadChannelTest, test Name= "<<testName <<std::endl;
	testsConfigured.push_back(testName);
	logFile<<"sonoqui0 "<< DeadChannelROOT::getAlgoName() <<std::endl;

	QCriterion * qc1 = mui->createQTest(DeadChannelROOT::getAlgoName(),testName);
	MEContentsXRangeROOT * me_qc1 = ( MEContentsXRangeROOT *) qc1;
	unsigned int threshold=(unsigned int)atof(params["threshold"].c_str());
	me_qc1->setMinimumEntries(7);
	//double warning=atof(params["warning"].c_str());
	//double error=atof(params["error"].c_str());
	//me_qc1->setWarningProb(warning);
	//me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableNoisyChannelTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){
	logFile<<"In QTestConfigure::EnableNoisyChannelTest, test Name= "<<testName <<std::endl;
	testsConfigured.push_back(testName);

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

void QTestConfigure::EnableMeanWithinExpectedTest(std::string testName, std::map<std::string, std::string> params, MonitorUserInterface * mui){
	
	logFile<<"In In QTestConfigure::EnableMeanWithinExpectedTest , test Name= "<<testName <<std::endl;
	testsConfigured.push_back(testName);
	QCriterion * qc1 = mui->createQTest(MeanWithinExpectedROOT::getAlgoName(),testName);
	MEMeanWithinExpectedROOT * me_qc1 = (MEMeanWithinExpectedROOT *) qc1;
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	double mean=atof(params["mean"].c_str());

	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
	me_qc1->setExpectedMean(mean);
	
	double useRMSVal=atof(params["useRMS"].c_str()); 
	double useSigmaVal=atof(params["useSigma"].c_str()); 
	double useRangeVal=atof(params["useRange"].c_str());
	if( useRMSVal&&useSigmaVal&&useRangeVal){
		std::cout<<"QTestConfigure::EnableMeanWithinExpectedTest: parameter set inconsistent, choose uniquely useRMS, useSigma, useRange."<<std::endl;
		return;
	}
	
	if(useRMSVal) {
		me_qc1->useRMS();
		return;
	}
	if(useSigmaVal) {
		me_qc1->useSigma(useSigmaVal);
		return;
	}	
	if(useRangeVal) {
		float xmin=atof(params["xmin"].c_str());
		float xmax=atof(params["xmax"].c_str());
		me_qc1->useRange(xmin,xmax);
		return;
	}	
	
	
}

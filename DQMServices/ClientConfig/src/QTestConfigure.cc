/** \file
 *
 *  Implementation of QTestConfigure
 *
 *  $Date: 2007/11/07 16:09:57 $
 *  $Revision: 1.8 $
 *  \author Ilaria Segoni
 */
#include "DQMServices/ClientConfig/interface/QTestConfigure.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

bool QTestConfigure::enableTests(std::map<std::string, std::map<std::string, std::string> > tests,DaqMonitorBEInterface *bei){
	
	testsConfigured.clear();
	
	std::map<std::string, std::map<std::string, std::string> >::iterator itr;
	for(itr= tests.begin(); itr!= tests.end();++itr){
 
		std::map<std::string, std::string> params= itr->second;
		
		std::string testName = itr->first; 
		std::string testType = params["type"]; 

		if(!std::strcmp(testType.c_str(),ContentsXRangeROOT::getAlgoName().c_str())) this->EnableXRangeTest(testName, params,bei);       
		if(!std::strcmp(testType.c_str(),ContentsYRangeROOT::getAlgoName().c_str())) this->EnableYRangeTest(testName, params,bei);       
		if(!std::strcmp(testType.c_str(),DeadChannelROOT::getAlgoName().c_str()))   this->EnableDeadChannelTest(testName, params,bei);       
		if(!std::strcmp(testType.c_str(),NoisyChannelROOT::getAlgoName().c_str()))  this->EnableNoisyChannelTest(testName, params,bei);       
		if(!std::strcmp(testType.c_str(),MeanWithinExpectedROOT::getAlgoName().c_str()))  this->EnableMeanWithinExpectedTest(testName, params,bei);       
                if(!std::strcmp(testType.c_str(),MostProbableLandauROOT::getAlgoName().c_str()))  this->EnableMostProbableLandauTest(testName, params, bei);

                if(!std::strcmp(testType.c_str(),ContentsTH2FWithinRangeROOT::getAlgoName().c_str())) this->EnableTH2FContentsInRangeTest(testName, params, bei);
                if(!std::strcmp(testType.c_str(),ContentsProfWithinRangeROOT::getAlgoName().c_str())) this->EnableProfContentsInRangeTest(testName, params, bei);
                if(!std::strcmp(testType.c_str(),ContentsProf2DWithinRangeROOT::getAlgoName().c_str())) this->EnableProf2DContentsInRangeTest(testName, params, bei);

	}
	
	return false;	
}



void QTestConfigure::EnableXRangeTest(std::string testName, std::map<std::string, std::string> params, DaqMonitorBEInterface *bei){
	QCriterion * qc1;	
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(ContentsXRangeROOT::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);
		
	}	
	MEContentsXRangeROOT * me_qc1 = (MEContentsXRangeROOT *) qc1;
	
	double xmin=atof(params["xmin"].c_str());
	double xmax=atof(params["xmax"].c_str());
	
	me_qc1->setAllowedXRange(xmin,xmax);
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableYRangeTest(std::string testName, std::map<std::string, std::string> params, DaqMonitorBEInterface *bei){
	QCriterion * qc1;	
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(ContentsYRangeROOT::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);	
	}	
	MEContentsYRangeROOT * me_qc1 = (MEContentsYRangeROOT *) qc1;
	
	double ymin=atof(params["ymin"].c_str());
	double ymax=atof(params["ymax"].c_str());
	me_qc1->setAllowedYRange(ymin,ymax);

	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableDeadChannelTest(std::string testName, std::map<std::string, std::string> params, DaqMonitorBEInterface *bei){
	QCriterion * qc1;
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(DeadChannelROOT::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);	
	
	}	
	MEDeadChannelROOT * me_qc1 = ( MEDeadChannelROOT *) qc1;
	
	unsigned int threshold=(unsigned int)atof(params["threshold"].c_str());
	me_qc1->setMinimumEntries(threshold);
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableNoisyChannelTest(std::string testName, std::map<std::string, std::string> params, DaqMonitorBEInterface *bei){

	QCriterion * qc1;
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(NoisyChannelROOT::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);			
	}
	MENoisyChannelROOT * me_qc1 = ( MENoisyChannelROOT *) qc1;
	
	unsigned int neighbors=(unsigned int)atof(params["neighbours"].c_str());
	double tolerance=atof(params["tolerance"].c_str());
	me_qc1->setNumNeighbors (neighbors);
  	me_qc1->setTolerance (tolerance);
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableMeanWithinExpectedTest(std::string testName, std::map<std::string, std::string> params, DaqMonitorBEInterface *bei){
	
	QCriterion * qc1;
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(MeanWithinExpectedROOT::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);				
	}
	MEMeanWithinExpectedROOT * me_qc1 = (MEMeanWithinExpectedROOT *) qc1;
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);

	double mean=atof(params["mean"].c_str());
	me_qc1->setExpectedMean(mean);
	
	double useRMSVal=atof(params["useRMS"].c_str()); 
	double useSigmaVal=atof(params["useSigma"].c_str()); 
	double useRangeVal=atof(params["useRange"].c_str());
	if( useRMSVal&&useSigmaVal&&useRangeVal){
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

void QTestConfigure::EnableMostProbableLandauTest( 
       const std::string                        &roTEST_NAME,
       std::map<std::string, std::string> &roMParams,
       DaqMonitorBEInterface                     *bei) {

  // Create QTest or Get already assigned one
  MEMostProbableLandauROOT *poQTest = 0;
  if( QCriterion *poQCriteration = bei->getQCriterion( roTEST_NAME)) {
    // Current already assigned to given ME.
    poQTest = dynamic_cast<MEMostProbableLandauROOT *>( poQCriteration);
  } else {
    // Test does not exist: create one
    testsConfigured.push_back( roTEST_NAME);
    poQCriteration = bei->createQTest( MostProbableLandauROOT::getAlgoName(),
                                         roTEST_NAME);

    poQTest = dynamic_cast<MEMostProbableLandauROOT *>( poQCriteration);
  }

  // Set probabilities thresholds.
  poQTest->setErrorProb    ( atof( roMParams["error"].c_str()) );
  poQTest->setWarningProb  ( atof( roMParams["warning"].c_str()) );
  poQTest->setXMin         ( atof( roMParams["xmin"].c_str()) );
  poQTest->setXMax         ( atof( roMParams["xmax"].c_str()) );
  poQTest->setNormalization( atof( roMParams["normalization"].c_str()) );
  poQTest->setMostProbable ( atof( roMParams["mostprobable"].c_str()) );
  poQTest->setSigma        ( atof( roMParams["sigma"].c_str()) );
}

void QTestConfigure::EnableTH2FContentsInRangeTest(std::string testName, std::map<std::string, std::string> params, DaqMonitorBEInterface *bei){

        QCriterion * qc1;
        if(! bei->getQCriterion(testName) ){
                testsConfigured.push_back(testName);
                qc1 = bei->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(),testName);
        }else{
                qc1 = bei->getQCriterion(testName);
        }
        MEContentsTH2FWithinRangeROOT * me_qc1 = (MEContentsTH2FWithinRangeROOT *) qc1;

        double warning=atof(params["warning"].c_str());
        double error=atof(params["error"].c_str());
        me_qc1->setWarningProb(warning);
        me_qc1->setErrorProb(error);

        double minMean=atof(params["minMean"].c_str());
        double maxMean=atof(params["maxMean"].c_str());
        if ( minMean != 0 || maxMean != 0 ) me_qc1->setMeanRange(minMean, maxMean);

        double minRMS=atof(params["minRMS"].c_str());
        double maxRMS=atof(params["maxRMS"].c_str());
        if ( minRMS != 0 || maxRMS != 0 ) me_qc1->setMeanRange(minRMS, maxRMS);

        double toleranceMean=atof(params["toleranceMean"].c_str());
        if ( toleranceMean != 0 ) me_qc1->setMeanTolerance(toleranceMean);

        int minEntries=atoi(params["minEntries"].c_str());
        if ( minEntries != 0 ) me_qc1->setMinimumEntries(minEntries);

}

void QTestConfigure::EnableProfContentsInRangeTest(std::string testName, std::map<std::string, std::string> params, DaqMonitorBEInterface *bei){

        QCriterion * qc1;
        if(! bei->getQCriterion(testName) ){
                testsConfigured.push_back(testName);
                qc1 = bei->createQTest(ContentsProfWithinRangeROOT::getAlgoName(),testName);
        }else{
                qc1 = bei->getQCriterion(testName);
        }
        MEContentsProfWithinRangeROOT * me_qc1 = (MEContentsProfWithinRangeROOT *) qc1;

        double warning=atof(params["warning"].c_str());
        double error=atof(params["error"].c_str());
        me_qc1->setWarningProb(warning);
        me_qc1->setErrorProb(error);

        double minMean=atof(params["minMean"].c_str());
        double maxMean=atof(params["maxMean"].c_str());
        if ( minMean != 0 || maxMean != 0 ) me_qc1->setMeanRange(minMean, maxMean);

        double minRMS=atof(params["minRMS"].c_str());
        double maxRMS=atof(params["maxRMS"].c_str());
        if ( minRMS != 0 || maxRMS != 0 ) me_qc1->setMeanRange(minRMS, maxRMS);

        double toleranceMean=atof(params["toleranceMean"].c_str());
        if ( toleranceMean != 0 ) me_qc1->setMeanTolerance(toleranceMean);

        int minEntries=atoi(params["minEntries"].c_str());
        if ( minEntries != 0 ) me_qc1->setMinimumEntries(minEntries);

}

void QTestConfigure::EnableProf2DContentsInRangeTest(std::string testName, std::map<std::string, std::string> params, DaqMonitorBEInterface *bei){

        QCriterion * qc1;
        if(! bei->getQCriterion(testName) ){
                testsConfigured.push_back(testName);
                qc1 = bei->createQTest(ContentsProf2DWithinRangeROOT::getAlgoName(),testName);
        }else{
                qc1 = bei->getQCriterion(testName);
        }
        MEContentsProf2DWithinRangeROOT * me_qc1 = (MEContentsProf2DWithinRangeROOT *) qc1;

        double warning=atof(params["warning"].c_str());
        double error=atof(params["error"].c_str());
        me_qc1->setWarningProb(warning);
        me_qc1->setErrorProb(error);

        double minMean=atof(params["minMean"].c_str());
        double maxMean=atof(params["maxMean"].c_str());
        if ( minMean != 0 || maxMean != 0 ) me_qc1->setMeanRange(minMean, maxMean);

        double minRMS=atof(params["minRMS"].c_str());
        double maxRMS=atof(params["maxRMS"].c_str());
        if ( minRMS != 0 || maxRMS != 0 ) me_qc1->setMeanRange(minRMS, maxRMS);

        double toleranceMean=atof(params["toleranceMean"].c_str());
        if ( toleranceMean != 0 ) me_qc1->setMeanTolerance(toleranceMean);

        int minEntries=atoi(params["minEntries"].c_str());
        if ( minEntries != 0 ) me_qc1->setMinimumEntries(minEntries);

}

void QTestConfigure::disableTests(std::vector<std::string> testsOFFList, DaqMonitorBEInterface *bei){
 std::vector<std::string>::iterator testsItr;
 for(testsItr= testsOFFList.begin(); testsItr != testsOFFList.end();++testsItr){ 
	if( bei->getQCriterion(*testsItr) ){
		QCriterion * qc1=bei->getQCriterion(*testsItr);
		qc1->disable();	
	}  
 }

}

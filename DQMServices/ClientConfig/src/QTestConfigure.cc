/** \file
 *
 *  Implementation of QTestConfigure
 *
 *  $Date: 2012/08/18 18:22:21 $
 *  $Revision: 1.26 $
 *  \author Ilaria Segoni
 */
#include "DQMServices/ClientConfig/interface/QTestConfigure.h"
#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <cstring>
#include <cstdlib>

bool QTestConfigure::enableTests(std::map<std::string, std::map<std::string, std::string> > tests,DQMStore *bei){
	
	testsConfigured.clear();
	
	std::map<std::string, std::map<std::string, std::string> >::iterator itr;
	for(itr= tests.begin(); itr!= tests.end();++itr){
 
		std::map<std::string, std::string> params= itr->second;
		
		std::string testName = itr->first; 
		std::string testType = params["type"]; 

		if(!std::strcmp(testType.c_str(),ContentsXRange::getAlgoName().c_str())) this->EnableXRangeTest(testName, params,bei);       
		if(!std::strcmp(testType.c_str(),ContentsYRange::getAlgoName().c_str())) this->EnableYRangeTest(testName, params,bei);       
		if(!std::strcmp(testType.c_str(),DeadChannel::getAlgoName().c_str()))   this->EnableDeadChannelTest(testName, params,bei);       
		if(!std::strcmp(testType.c_str(),NoisyChannel::getAlgoName().c_str()))  this->EnableNoisyChannelTest(testName, params,bei);       
		if(!std::strcmp(testType.c_str(),MeanWithinExpected::getAlgoName().c_str()))  this->EnableMeanWithinExpectedTest(testName, params,bei);     

		//================================== new qtests in the parser =============================================================//
                if(!std::strcmp(testType.c_str(),Comp2RefEqualH::getAlgoName().c_str())) this->EnableComp2RefEqualHTest(testName, params,bei);
                if(!std::strcmp(testType.c_str(),  Comp2RefChi2::getAlgoName().c_str())) this->EnableComp2RefChi2Test(testName, params,bei); 
                if(!std::strcmp(testType.c_str(),Comp2RefKolmogorov::getAlgoName().c_str())) this->EnableComp2RefKolmogorovTest(testName, params,bei);

  
/*
                if(!std::strcmp(testType.c_str(),MostProbableLandau::getAlgoName().c_str()))  this->EnableMostProbableLandauTest(testName, params, bei);
*/

                if(!std::strcmp(testType.c_str(),ContentsWithinExpected::getAlgoName().c_str())) this->EnableContentsWithinExpectedTest(testName, params, bei);
//              if(!std::strcmp(testType.c_str(),ContentsWithinExpectedAS::getAlgoName().c_str())) this->EnableContentsWithinExpectedASTest(testName, params, bei);

                if(!std::strcmp(testType.c_str(),CompareToMedian::getAlgoName().c_str())) this->EnableCompareToMedianTest(testName, params, bei);
                if(!std::strcmp(testType.c_str(),CompareLastFilledBin::getAlgoName().c_str())) this->EnableCompareLastFilledBinTest(testName, params, bei);

	}
	
	return false;	
}



void QTestConfigure::EnableComp2RefEqualHTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){
	QCriterion * qc1;	
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(Comp2RefEqualH::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);
		
	}	
	Comp2RefEqualH * me_qc1 = (Comp2RefEqualH *) qc1;
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}



void QTestConfigure::EnableComp2RefChi2Test(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){
	QCriterion * qc1;	
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(Comp2RefChi2::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);
		
	}	
	Comp2RefChi2 * me_qc1 = (Comp2RefChi2 *) qc1;
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}


void QTestConfigure::EnableComp2RefKolmogorovTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){
	QCriterion * qc1;	
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(Comp2RefKolmogorov::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);
		
	}	
	Comp2RefKolmogorov * me_qc1 = (Comp2RefKolmogorov *) qc1;
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableXRangeTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){
	QCriterion * qc1;	
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(ContentsXRange::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);
		
	}	
	ContentsXRange * me_qc1 = (ContentsXRange *) qc1;
	
	double xmin=atof(params["xmin"].c_str());
	double xmax=atof(params["xmax"].c_str());
	
	me_qc1->setAllowedXRange(xmin,xmax);
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}


void QTestConfigure::EnableYRangeTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){
	QCriterion * qc1;	
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(ContentsYRange::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);	
	}	
	ContentsYRange * me_qc1 = (ContentsYRange *) qc1;
	
	double ymin=atof(params["ymin"].c_str());
	double ymax=atof(params["ymax"].c_str());
	me_qc1->setAllowedYRange(ymin,ymax);

        //do a Normal test or AS ?
        unsigned int useEmptyBins=(unsigned int)atof(params["useEmptyBins"].c_str());	
	me_qc1->setUseEmptyBins(useEmptyBins);

	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableDeadChannelTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){
	QCriterion * qc1;
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(DeadChannel::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);	
	
	}	
	DeadChannel * me_qc1 = ( DeadChannel *) qc1;
	
	unsigned int threshold=(unsigned int)atof(params["threshold"].c_str());
	me_qc1->setThreshold(threshold);
      
        // Include/Exclude empty bins ?
        //unsigned int useEmptyBins=(unsigned int)atof(params["useEmptyBins"].c_str());
        //me_qc1->setUseEmptyBins(useEmptyBins);
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableNoisyChannelTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){

	QCriterion * qc1;
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(NoisyChannel::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);			
	}
	NoisyChannel * me_qc1 = ( NoisyChannel *) qc1;
	
	unsigned int neighbors=(unsigned int)atof(params["neighbours"].c_str());
	double tolerance=atof(params["tolerance"].c_str());
	me_qc1->setNumNeighbors (neighbors);
  	me_qc1->setTolerance (tolerance);
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableMeanWithinExpectedTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){
	
	QCriterion * qc1;
  	if(! bei->getQCriterion(testName) ){
		testsConfigured.push_back(testName);
		qc1 = bei->createQTest(MeanWithinExpected::getAlgoName(),testName);
	}else{
		qc1 = bei->getQCriterion(testName);				
	}
	MeanWithinExpected * me_qc1 = (MeanWithinExpected *) qc1;
	
	double warning=atof(params["warning"].c_str());
	double error=atof(params["error"].c_str());
	me_qc1->setWarningProb(warning);
	me_qc1->setErrorProb(error);

	double mean=atof(params["mean"].c_str());
	me_qc1->setExpectedMean(mean);
	
        int minEntries=atoi(params["minEntries"].c_str());
        if ( minEntries != 0 ) me_qc1->setMinimumEntries(minEntries);

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

void QTestConfigure::EnableCompareToMedianTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei)
{
  QCriterion *qc1;
  if(! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(CompareToMedian::getAlgoName(),testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  CompareToMedian* vtm = (CompareToMedian*) qc1;
  vtm->setMin((double)atof(params["MinRel"].c_str()));
  vtm->setMax((double)atof(params["MaxRel"].c_str()));
  vtm->setEmptyBins((int)atoi(params["UseEmptyBins"].c_str()));
  vtm->setMinMedian((double)atof(params["MinAbs"].c_str()));
  vtm->setMaxMedian((double)atof(params["MaxAbs"].c_str()));
  vtm->setWarningProb((double)atof(params["warning"].c_str()));
  vtm->setErrorProb((double)atof(params["error"].c_str()));
  vtm->setStatCut((double)atof(params["StatCut"].c_str()));
}

/*
void QTestConfigure::EnableMostProbableLandauTest( 
       const std::string                        &roTEST_NAME,
       std::map<std::string, std::string> &roMParams,
       DQMStore                     *bei) {

  // Create QTest or Get already assigned one
  MostProbableLandau *poQTest = 0;
  if( QCriterion *poQCriteration = bei->getQCriterion( roTEST_NAME)) {
    // Current already assigned to given ME.
    poQTest = dynamic_cast<MostProbableLandau *>( poQCriteration);
  } else {
    // Test does not exist: create one
    testsConfigured.push_back( roTEST_NAME);
    poQCriteration = bei->createQTest( MostProbableLandau::getAlgoName(),
                                         roTEST_NAME);

    poQTest = dynamic_cast<MostProbableLandau *>( poQCriteration);
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
*/
void QTestConfigure::EnableContentsWithinExpectedTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){

        QCriterion * qc1;
        if(! bei->getQCriterion(testName) ){
                testsConfigured.push_back(testName);
                qc1 = bei->createQTest(ContentsWithinExpected::getAlgoName(),testName);
        }else{
                qc1 = bei->getQCriterion(testName);
        }
        ContentsWithinExpected * me_qc1 = (ContentsWithinExpected *) qc1;


        //do a Normal test or AS ?
        unsigned int useEmptyBins=(unsigned int)atof(params["useEmptyBins"].c_str());	
	me_qc1->setUseEmptyBins(useEmptyBins);

        double warning=atof(params["warning"].c_str());
        double error=atof(params["error"].c_str());
        me_qc1->setWarningProb(warning);
        me_qc1->setErrorProb(error);

        double minMean=atof(params["minMean"].c_str());
        double maxMean=atof(params["maxMean"].c_str());
        if ( minMean != 0 || maxMean != 0 ) me_qc1->setMeanRange(minMean, maxMean);

        double minRMS=atof(params["minRMS"].c_str());
        double maxRMS=atof(params["maxRMS"].c_str());
        if ( minRMS != 0 || maxRMS != 0 ) me_qc1->setRMSRange(minRMS, maxRMS);

        double toleranceMean=atof(params["toleranceMean"].c_str());
        if ( toleranceMean != 0 ) me_qc1->setMeanTolerance(toleranceMean);

        int minEntries=atoi(params["minEntries"].c_str());
        if ( minEntries != 0 ) me_qc1->setMinimumEntries(minEntries);
}
void QTestConfigure::EnableCompareLastFilledBinTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){

        QCriterion * qc1;
        if(! bei->getQCriterion(testName) ){
                testsConfigured.push_back(testName);
                qc1 = bei->createQTest(CompareLastFilledBin::getAlgoName(),testName);
        }else{
                qc1 = bei->getQCriterion(testName);
        }
        CompareLastFilledBin * me_qc1 = (CompareLastFilledBin *) qc1;

        double warning=atof(params["warning"].c_str());
        double error=atof(params["error"].c_str());
        me_qc1->setWarningProb(warning);
        me_qc1->setErrorProb(error);

        double avVal=atof(params["AvVal"].c_str());
        me_qc1->setAverage(avVal);

        double minVal=atof(params["MinVal"].c_str());
        me_qc1->setMin(minVal);

        double maxVal=atof(params["MaxVal"].c_str());
        me_qc1->setMax(maxVal);
}

/* void QTestConfigure::EnableContentsWithinExpectedASTest(std::string testName, std::map<std::string, std::string> params, DQMStore *bei){

        QCriterion * qc1;
        if(! bei->getQCriterion(testName) ){
                testsConfigured.push_back(testName);
                qc1 = bei->createQTest(ContentsWithinExpectedAS::getAlgoName(),testName);
        }else{
                qc1 = bei->getQCriterion(testName);
        }
        ContentsWithinExpectedAS * me_qc1 = (ContentsWithinExpectedAS *) qc1;

        double warning=atof(params["warning"].c_str());
        double error=atof(params["error"].c_str());
        me_qc1->setWarningProb(warning);
        me_qc1->setErrorProb(error);

        double minCont=atof(params["minCont"].c_str());
        double maxCont=atof(params["maxCont"].c_str());
        if ( minCont != 0 || maxCont != 0 ) me_qc1->setContentsRange(minCont, maxCont);


} */


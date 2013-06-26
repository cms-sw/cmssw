/** \file
 *
 *  Implementation of QTestConfigure
 *
 *  $Date: 2013/05/30 15:29:53 $
 *  $Revision: 1.28 $
 *  \author Ilaria Segoni
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/ClientConfig/interface/QTestConfigure.h"
#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <cstring>
#include <cstdlib>


const char * QTestConfigure::findOrDefault(const std::map<std::string, std::string> &m,
                                           const char * item,
                                           const char * default_value) const {
  std::map<std::string, std::string>::const_iterator iter;
  if (( iter = m.find(std::string(item))) != m.end()) {
    return (*iter).second.c_str();
  }
  edm::LogWarning("QTestConfigure") << "Warning, using default value for parameter "
                                    << item << " with default_value: "
                                    << default_value << std::endl;
  return default_value;
}

bool QTestConfigure::enableTests(
    const std::map<std::string, std::map<std::string, std::string> > & tests,
    DQMStore * bei) {
  testsConfigured.clear();
  std::map<std::string, std::map<std::string, std::string> >::const_iterator itr;
  for (itr = tests.begin(); itr!= tests.end();++itr) {
    const std::map<std::string, std::string> &params= itr->second;

    std::string testName = itr->first;
    std::string testType = params.at("type");

    if(!std::strcmp(testType.c_str(),ContentsXRange::getAlgoName().c_str()))
      this->EnableXRangeTest(testName, params,bei);
    if(!std::strcmp(testType.c_str(),ContentsYRange::getAlgoName().c_str()))
      this->EnableYRangeTest(testName, params,bei);
    if(!std::strcmp(testType.c_str(),DeadChannel::getAlgoName().c_str()))
      this->EnableDeadChannelTest(testName, params,bei);
    if(!std::strcmp(testType.c_str(),NoisyChannel::getAlgoName().c_str()))
      this->EnableNoisyChannelTest(testName, params,bei);
    if(!std::strcmp(testType.c_str(),MeanWithinExpected::getAlgoName().c_str()))
      this->EnableMeanWithinExpectedTest(testName, params,bei);
    if(!std::strcmp(testType.c_str(),Comp2RefEqualH::getAlgoName().c_str()))
      this->EnableComp2RefEqualHTest(testName, params,bei);
    if(!std::strcmp(testType.c_str(),  Comp2RefChi2::getAlgoName().c_str()))
      this->EnableComp2RefChi2Test(testName, params,bei);
    if(!std::strcmp(testType.c_str(),Comp2RefKolmogorov::getAlgoName().c_str()))
      this->EnableComp2RefKolmogorovTest(testName, params,bei);
    if(!std::strcmp(testType.c_str(),ContentsWithinExpected::getAlgoName().c_str()))
      this->EnableContentsWithinExpectedTest(testName, params, bei);
    if(!std::strcmp(testType.c_str(),CompareToMedian::getAlgoName().c_str()))
      this->EnableCompareToMedianTest(testName, params, bei);
    if(!std::strcmp(testType.c_str(),CompareLastFilledBin::getAlgoName().c_str()))
      this->EnableCompareLastFilledBinTest(testName, params, bei);

  }
  return false;
}

void QTestConfigure::EnableComp2RefEqualHTest(std::string testName,
                                              const std::map<std::string, std::string> & params,
                                              DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(Comp2RefEqualH::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  Comp2RefEqualH * me_qc1 = (Comp2RefEqualH *) qc1;
  double warning = atof(findOrDefault(params, "warning", "0"));
  double error   = atof(findOrDefault(params, "error", "0"));
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableComp2RefChi2Test(std::string testName,
                                            const std::map<std::string, std::string> & params,
                                            DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(Comp2RefChi2::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  Comp2RefChi2 * me_qc1 = (Comp2RefChi2 *) qc1;
  double warning = atof(findOrDefault(params, "warning", "0"));
  double error   = atof(findOrDefault(params, "error", "0"));
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);
}


void QTestConfigure::EnableComp2RefKolmogorovTest(std::string testName,
                                                  const std::map<std::string, std::string> & params,
                                                  DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(Comp2RefKolmogorov::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  Comp2RefKolmogorov * me_qc1 = (Comp2RefKolmogorov *) qc1;
  double warning = atof(findOrDefault(params, "warning", "0"));
  double error   = atof(findOrDefault(params, "error", "0"));
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableXRangeTest(std::string testName,
                                      const std::map<std::string, std::string> & params,
                                      DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(ContentsXRange::getAlgoName(),testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  ContentsXRange * me_qc1 = (ContentsXRange *) qc1;
  double xmin    = atof(findOrDefault(params, "xmin", "0"));
  double xmax    = atof(findOrDefault(params, "xmax", "0"));
  double warning = atof(findOrDefault(params, "warning", "0"));
  double error   = atof(findOrDefault(params, "error", "0"));
  me_qc1->setAllowedXRange(xmin, xmax);
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableYRangeTest(std::string testName,
                                      const std::map<std::string, std::string> & params,
                                      DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(ContentsYRange::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  ContentsYRange * me_qc1 = (ContentsYRange *) qc1;
  double ymin    = atof(findOrDefault(params, "ymin", "0"));
  double ymax    = atof(findOrDefault(params, "ymax", "0"));
  double warning = atof(findOrDefault(params, "warning", "0"));
  double error   = atof(findOrDefault(params, "error", "0"));
  unsigned int useEmptyBins = (unsigned int)atof(findOrDefault(params, "useEmptyBins", "0"));
  me_qc1->setAllowedYRange(ymin, ymax);
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);
  me_qc1->setUseEmptyBins(useEmptyBins);
}

void QTestConfigure::EnableDeadChannelTest(std::string testName,
                                           const std::map<std::string, std::string> & params,
                                           DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(DeadChannel::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  DeadChannel * me_qc1 = ( DeadChannel *) qc1;
  unsigned int threshold =(unsigned int) atof(findOrDefault(params, "threshold", "0"));
  double warning = atof(findOrDefault(params, "warning", "0"));
  double error   = atof(findOrDefault(params, "error", "0"));
  me_qc1->setThreshold(threshold);
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableNoisyChannelTest(std::string testName,
                                            const std::map<std::string, std::string> & params,
                                            DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(NoisyChannel::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  NoisyChannel * me_qc1  = (NoisyChannel *) qc1;
  unsigned int neighbors = (unsigned int) atof(findOrDefault(params, "neighbours", "0"));
  double tolerance = atof(findOrDefault(params, "tolerance", "0"));
  double warning   = atof(findOrDefault(params, "warning", "0"));
  double error     = atof(findOrDefault(params, "error", "0"));
  me_qc1->setNumNeighbors (neighbors);
  me_qc1->setTolerance (tolerance);
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);
}

void QTestConfigure::EnableMeanWithinExpectedTest(std::string testName,
                                                  const std::map<std::string, std::string> & params,
                                                  DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(MeanWithinExpected::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  MeanWithinExpected * me_qc1 = (MeanWithinExpected *) qc1;
  double warning     = atof(findOrDefault(params, "warning", "0"));
  double error       = atof(findOrDefault(params, "error", "0"));
  double mean        = atof(findOrDefault(params, "mean", "0"));
  int minEntries     = atoi(findOrDefault(params, "minEntries", "0"));
  double useRMSVal   = atof(findOrDefault(params, "useRMS", "0"));
  double useSigmaVal = atof(findOrDefault(params, "useSigma", "0"));
  double useRangeVal = atof(findOrDefault(params, "useRange", "0"));
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);
  me_qc1->setExpectedMean(mean);

  if (minEntries != 0)
    me_qc1->setMinimumEntries(minEntries);

  if (useRMSVal && useSigmaVal && useRangeVal)
    return;

  if (useRMSVal) {
    me_qc1->useRMS();
    return;
  }

  if (useSigmaVal) {
    me_qc1->useSigma(useSigmaVal);
    return;
  }

  if(useRangeVal) {
    float xmin = atof(findOrDefault(params, "xmin", "0"));
    float xmax = atof(findOrDefault(params, "xmax", "0"));
    me_qc1->useRange(xmin,xmax);
    return;
  }
}

void QTestConfigure::EnableCompareToMedianTest(std::string testName,
                                               const std::map<std::string, std::string> & params,
                                               DQMStore *bei) {
  QCriterion *qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(CompareToMedian::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  CompareToMedian* vtm = (CompareToMedian*) qc1;
  vtm->setMin( (double) atof(findOrDefault(params, "MinRel", "0")));
  vtm->setMax( (double) atof(findOrDefault(params, "MaxRel", "0")));
  vtm->setEmptyBins( (int) atoi(findOrDefault(params, "UseEmptyBins", "0")));
  vtm->setMinMedian( (double) atof(findOrDefault(params, "MinAbs", "0")));
  vtm->setMaxMedian( (double) atof(findOrDefault(params, "MaxAbs", "0")));
  vtm->setWarningProb( (double) atof(findOrDefault(params, "warning", "0")));
  vtm->setErrorProb( (double) atof(findOrDefault(params, "error", "0")));
  vtm->setStatCut( (double) atof(findOrDefault(params, "StatCut", "0")));
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
  poQTest->setErrorProb    ( atof( roMfindOrDefault(params, "error", "0") );
  poQTest->setWarningProb  ( atof( roMfindOrDefault(params, "warning", "0") );
  poQTest->setXMin         ( atof( roMfindOrDefault(params, "xmin", "0") );
  poQTest->setXMax         ( atof( roMfindOrDefault(params, "xmax", "0") );
  poQTest->setNormalization( atof( roMfindOrDefault(params, "normalization", "0") );
  poQTest->setMostProbable ( atof( roMfindOrDefault(params, "mostprobable", "0") );
  poQTest->setSigma        ( atof( roMfindOrDefault(params, "sigma", "0") );
  }
*/
void QTestConfigure::EnableContentsWithinExpectedTest(std::string testName,
                                                      const std::map<std::string, std::string> & params,
                                                      DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(ContentsWithinExpected::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  ContentsWithinExpected * me_qc1 = (ContentsWithinExpected *) qc1;
  unsigned int useEmptyBins = (unsigned int) atof(findOrDefault(params, "useEmptyBins", "0"));
  double warning       = atof(findOrDefault(params, "warning", "0"));
  double error         = atof(findOrDefault(params, "error", "0"));
  double minMean       = atof(findOrDefault(params, "minMean", "0"));
  double maxMean       = atof(findOrDefault(params, "maxMean", "0"));
  double minRMS        = atof(findOrDefault(params, "minRMS", "0"));
  double maxRMS        = atof(findOrDefault(params, "maxRMS", "0"));
  double toleranceMean = atof(findOrDefault(params, "toleranceMean", "0"));
  int minEntries       = atoi(findOrDefault(params, "minEntries", "0"));
  me_qc1->setUseEmptyBins(useEmptyBins);
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);

  if ( minMean != 0 || maxMean != 0 )
    me_qc1->setMeanRange(minMean, maxMean);

  if ( minRMS != 0 || maxRMS != 0 )
    me_qc1->setRMSRange(minRMS, maxRMS);

  if ( toleranceMean != 0 )
    me_qc1->setMeanTolerance(toleranceMean);

  if ( minEntries != 0 )
    me_qc1->setMinimumEntries(minEntries);
}

void QTestConfigure::EnableCompareLastFilledBinTest(std::string testName,
                                                    const std::map<std::string, std::string> & params,
                                                    DQMStore *bei) {
  QCriterion * qc1;
  if (! bei->getQCriterion(testName)) {
    testsConfigured.push_back(testName);
    qc1 = bei->createQTest(CompareLastFilledBin::getAlgoName(), testName);
  } else {
    qc1 = bei->getQCriterion(testName);
  }
  CompareLastFilledBin * me_qc1 = (CompareLastFilledBin *) qc1;

  double warning = atof(findOrDefault(params, "warning", "0"));
  double error   = atof(findOrDefault(params, "error", "0"));
  double avVal   = atof(findOrDefault(params, "AvVal", "0"));
  double minVal  = atof(findOrDefault(params, "MinVal", "0"));
  double maxVal  = atof(findOrDefault(params, "MaxVal", "0"));
  me_qc1->setWarningProb(warning);
  me_qc1->setErrorProb(error);
  me_qc1->setAverage(avVal);
  me_qc1->setMin(minVal);
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

double warning=atof(findOrDefault(params, "warning", "0");
double error=atof(findOrDefault(params, "error", "0");
me_qc1->setWarningProb(warning);
me_qc1->setErrorProb(error);

double minCont=atof(findOrDefault(params, "minCont", "0");
double maxCont=atof(findOrDefault(params, "maxCont", "0");
if ( minCont != 0 || maxCont != 0 ) me_qc1->setContentsRange(minCont, maxCont);


} */


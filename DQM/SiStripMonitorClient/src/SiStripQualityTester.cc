/** \file
 *
 *  Implementation of SiStripQualityTester
 *
 *  $Date: 2006/11/13 21:55:38 $
 *  $Revision: 1.12 $
 *  \author Suchandra Dutta
 */
#include "DQM/SiStripMonitorClient/interface/SiStripQualityTester.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/ClientConfig/interface/QTestNames.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include<iostream>
#include <fstream>
using namespace std;
//
// -- Constructor
//
SiStripQualityTester::SiStripQualityTester() {
  edm::LogInfo("SiStripQualityTester") << 
    " Creating SiStripQualityTester " << "\n" ;
  theQTestMap.clear();
  theMeAssociateMap.clear();
}
//
// -- Destructor
//
SiStripQualityTester::~SiStripQualityTester() {
     edm::LogInfo("SiStripQualityTester") << 
	" Deleting SiStripQualityTester " << "\n";
  theQTestMap.clear();
  theMeAssociateMap.clear();
}
//
// -- Set up Quality Tests 
//
void SiStripQualityTester::setupQTests(MonitorUserInterface* mui) {
  string localPath = string("DQM/SiStripMonitorClient/test/sistrip_qualitytest_config.xml");
  if (theQTestMap.size() == 0) {
    readQualityTests(edm::FileInPath(localPath).fullPath());
  }
  for (SiStripQualityTester::QTestMapType::iterator it = theQTestMap.begin();
       it != theQTestMap.end(); it++) {
    string qTestName = it->first;
    map<string, string> qTestParams = it->second;
    string qTestType = qTestParams[dqm::qtest_config::type];
    if (qTestType == dqm::qtest_config::XRangeContent) {
      setXRangeTest(mui, qTestName, qTestParams);
    } else if (qTestType == "MeanWithinExpectedROOT") {
      setMeanWithinExpectedTest(mui, qTestName, qTestParams);
    }
  }
  cout << " Attaching Quality Tests " << endl;
  
  attachTests(mui);
  cout <<  " Quality Tests attached to MEs properly" << endl;
}
//
// -- Read Test Name and Parameters
//
void SiStripQualityTester::readQualityTests(string fname) {
  // Instantiate the parser and read tests 
  QTestConfigurationParser qTestParser;
  qTestParser.getDocument(fname);
  qTestParser.parseQTestsConfiguration();
  theQTestMap = qTestParser.testsList();
  theMeAssociateMap =  qTestParser.meToTestsList();
}  
//
//
// -- Attach Quality Tests to ME's
//
void SiStripQualityTester::attachTests(MonitorUserInterface * mui){
  string currDir = mui->pwd();
  vector<string> contentVec;
  mui->getContents(contentVec);
  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    vector<string> contents;
    int nval = SiStripUtility::getMEList((*it), contents);
    if (nval == 0) continue;
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
    
      for (SiStripQualityTester::MEAssotiateMapType::const_iterator ic = theMeAssociateMap.begin(); ic != theMeAssociateMap.end(); ic++) {
        string me_name = ic->first;
        if ((*im).find(me_name) != string::npos) {
	  for (vector<string>::const_iterator iv = ic->second.begin();
	       iv !=  ic->second.end(); iv++) {
	    mui->useQTest((*im), (*iv));
	  }
        }
      }
    }
  }
}
//
// -- Set up ContentsXRange test with it's parameters
//
void SiStripQualityTester::setXRangeTest(MonitorUserInterface * mui, 
                   string name, map<string, string>& params){
  
  QCriterion* qc = mui->createQTest(ContentsXRangeROOT::getAlgoName(),name);

  MEContentsXRangeROOT * me_qc = dynamic_cast<MEContentsXRangeROOT *> (qc);
  if (params.size() < 4) return;
  cout << " Setting Parameters for " <<ContentsXRangeROOT::getAlgoName()
       << endl; 
  // set allowed range in X-axis (default values: histogram's X-range)
  float xmin =  atof(params["xmin"].c_str());
  float xmax =  atof(params["xmax"].c_str());
  me_qc->setAllowedXRange(xmin, xmax);
  //set probability limit for test warning 
  me_qc->setWarningProb(atof(params["warning"].c_str()));
  //set probability limit for test error 
  me_qc->setErrorProb(atof(params["error"].c_str()));
}
//
// -- Set up MeanWithinExpected test with it's parameters
//
void SiStripQualityTester::setMeanWithinExpectedTest(MonitorUserInterface* mui,
                          string name, map<string, string>& params){
  QCriterion* qc = mui->createQTest(MeanWithinExpectedROOT::getAlgoName(),name);
  MEMeanWithinExpectedROOT * me_qc = dynamic_cast<MEMeanWithinExpectedROOT *> (qc);
  if (params.size() < 6 ) return;
  cout << " Setting Parameters for " <<MeanWithinExpectedROOT::getAlgoName()
       << endl; 
  //set probability limit for test warning
  me_qc->setWarningProb(atof(params["warning"].c_str()));
  //set probability limit for test error
  me_qc->setErrorProb(atof(params["error"].c_str()));
  // set Expected Mean
  me_qc->setExpectedMean(atof(params["mean"].c_str()));
  // set Test Type
  if (params["useRMS"] == "1") me_qc->useRMS();
  else if (params["useSigma"] != "0") me_qc->useSigma(atof(params["useSigma"].c_str()));
}
//
// -- Get names of MEs under Quality Test
//
int SiStripQualityTester::getMEsUnderTest(vector<string>& me_names){  
  if (theMeAssociateMap.size() == 0) {
    cout << " SiStripQualityTester::getMEsUnderTest ==> " << 
      " ME association Map emptyis empty!!! " << endl;
    return 0;
  }
  me_names.clear();
  for (SiStripQualityTester::MEAssotiateMapType::const_iterator 
	 it = theMeAssociateMap.begin(); it != theMeAssociateMap.end(); it++) {
    me_names.push_back(it->first);
  }
  return me_names.size();
}

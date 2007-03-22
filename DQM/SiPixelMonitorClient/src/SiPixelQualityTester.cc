/** \file
 *
 *  Implementation of SiPixelQualityTester
 *
 *  $Date: 2007/02/16 15:45:00 $
 *  $Revision: 1.3 $
 *  \author Petra Merkel
 */
#include "DQM/SiPixelMonitorClient/interface/SiPixelQualityTester.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/ClientConfig/interface/QTestNames.h"

#include<iostream>
#include <fstream>
using namespace std;
//
// -- Constructor
//
SiPixelQualityTester::SiPixelQualityTester() {
  edm::LogInfo("SiPixelQualityTester") << 
    " Creating SiPixelQualityTester " << "\n" ;
  theQTestMap.clear();
  theMeAssociateMap.clear();
}
//
// -- Destructor
//
SiPixelQualityTester::~SiPixelQualityTester() {
     edm::LogInfo("SiPixelQualityTester") << 
	" Deleting SiPixelQualityTester " << "\n";
  theQTestMap.clear();
  theMeAssociateMap.clear();
}
//
// -- Set up Quality Tests 
//
void SiPixelQualityTester::setupQTests(MonitorUserInterface* mui) {
  if (theQTestMap.size() == 0) {
    readQualityTests("sipixel_qualitytest_config.xml");
    cout<<"how many qtests found?"<<theQTestMap.size()<<endl;
  }
  for (SiPixelQualityTester::QTestMapType::iterator it = theQTestMap.begin();
       it != theQTestMap.end(); it++) {
    string qTestName = it->first;
    map<string, string> qTestParams = it->second;
    string qTestType = qTestParams[dqm::qtest_config::type];
    if (qTestType == "ContentsXRange") {
      setXRangeTest(mui, qTestName, qTestParams);
    } else if (qTestType == "ContentsYRange") {
      setYRangeTest(mui, qTestName, qTestParams);
    } else if (qTestType == "MeanWithinExpected") {
      setMeanWithinExpectedTest(mui, qTestName, qTestParams);
    }
  }
//  cout << " Attaching Quality Tests " << endl;
  
  attachTests(mui);
//  cout <<  " Quality Tests attached to MEs properly" << endl;
}
//
// -- Read Test Name and Parameters
//
void SiPixelQualityTester::readQualityTests(string fname) {
  // Instantiate the parser and read tests 
  QTestConfigurationParser qTestParser;
  qTestParser.getDocument(fname);
  bool fetchfile = qTestParser.parseQTestsConfiguration();
  cout<<"file opened ok? "<<fetchfile<<"(if 0, no errors.)"<<endl;
  qTestParser.parseQTestsConfiguration();
  theQTestMap = qTestParser.testsList();
  theMeAssociateMap =  qTestParser.meToTestsList();
}  
//
//
// -- Attach Quality Tests to ME's
//
void SiPixelQualityTester::attachTests(MonitorUserInterface * mui){
  string currDir = mui->pwd();
  vector<string> contentVec;
  mui->getContents(contentVec);
  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    vector<string> contents;
    int nval = SiPixelUtility::getMEList((*it), contents);
    if (nval == 0) continue;
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
    
      for (SiPixelQualityTester::MEAssotiateMapType::const_iterator ic = theMeAssociateMap.begin(); ic != theMeAssociateMap.end(); ic++) {
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
void SiPixelQualityTester::setXRangeTest(MonitorUserInterface * mui, 
                   string name, map<string, string>& params){
  
  QCriterion* qc = mui->createQTest(ContentsXRangeROOT::getAlgoName(),name);

  MEContentsXRangeROOT * me_qc = (MEContentsXRangeROOT *) qc;
  if (params.size() < 4) return;
//  cout << " Setting Parameters for " <<ContentsXRangeROOT::getAlgoName() << endl; 
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
// -- Set up ContentsYRange test with it's parameters
//
void SiPixelQualityTester::setYRangeTest(MonitorUserInterface * mui, 
                   string name, map<string, string>& params){
  
  QCriterion* qc = mui->createQTest(ContentsYRangeROOT::getAlgoName(),name);

  MEContentsYRangeROOT * me_qc = (MEContentsYRangeROOT *) qc;
  if (params.size() < 4) return;
//  cout << " Setting Parameters for " <<ContentsXRangeROOT::getAlgoName() << endl; 
  // set allowed range in X-axis (default values: histogram's X-range)
  float ymin =  atof(params["ymin"].c_str());
  float ymax =  atof(params["ymax"].c_str());
  me_qc->setAllowedYRange(ymin, ymax);
  //set probability limit for test warning 
  me_qc->setWarningProb(atof(params["warning"].c_str()));
  //set probability limit for test error 
  me_qc->setErrorProb(atof(params["error"].c_str()));
}
//
// -- Set up MeanWithinExpected test with it's parameters
//
void SiPixelQualityTester::setMeanWithinExpectedTest(MonitorUserInterface* mui,
                          string name, map<string, string>& params){
  QCriterion* qc = mui->createQTest(MeanWithinExpectedROOT::getAlgoName(),name);
  MEMeanWithinExpectedROOT * me_qc = (MEMeanWithinExpectedROOT *) qc;
  if (params.size() < 6 ) return;
//  cout << " Setting Parameters for " <<MeanWithinExpectedROOT::getAlgoName() << endl; 
  //set probability limit for test warning
  me_qc->setWarningProb(atof(params["warning"].c_str()));
  //set probability limit for test error
  me_qc->setErrorProb(atof(params["error"].c_str()));
  // set Expected Mean
  me_qc->setExpectedMean(atof(params["mean"].c_str()));
  float xmin =  atof(params["xmin"].c_str());
  float xmax =  atof(params["xmax"].c_str());
  // set Test Type
  if (params["useRMS"] == "1") me_qc->useRMS();
  else if (params["useSigma"] != "0") me_qc->useSigma(atof(params["useSigma"].c_str()));
  else if (params["useRange"] != "0") me_qc->useRange(xmin,xmax);
}
//
// -- Get names of MEs under Quality Test
//
int SiPixelQualityTester::getMEsUnderTest(vector<string>& me_names){  
  if (theMeAssociateMap.size() == 0) {
    cout << " SiPixelQualityTester::getMEsUnderTest ==> " << 
      " ME association Map is empty!!! " << endl;
    return 0;
  }
  me_names.clear();
  for (SiPixelQualityTester::MEAssotiateMapType::const_iterator 
	 it = theMeAssociateMap.begin(); it != theMeAssociateMap.end(); it++) {
    me_names.push_back(it->first);
  }
  return me_names.size();
}

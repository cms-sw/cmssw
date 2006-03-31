/** \file
 *
 *  Implementation of SiStripQualityTester
 *
 *  $Date: 2006/03/01 15:50:32 $
 *  $Revision: 1.0 $
 *  \author Suchandra Dutta
 */
#include "DQM/SiStripMonitorClient/interface/SiStripQualityTester.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>
#include <fstream>
//
// -- Constructor
//
SiStripQualityTester::SiStripQualityTester() {
  edm::LogInfo("SiStripQualityTester") << 
    " Creating SiStripQualityTester " << "\n" ;
}
//
// -- Destructor
//
SiStripQualityTester::~SiStripQualityTester() {
     edm::LogInfo("SiStripQualityTester") << 
	" Deleting SiStripQualityTester " << "\n";
}
//
// -- Set up Quality Tests 
//
void SiStripQualityTester::setupQTests( MonitorUserInterface* mui) {
  if (theQTestMap.size() == 0) readFromFile("test.txt");
  for (SiStripQualityTester::TestMapType::iterator it = theQTestMap.begin();
       it != theQTestMap.end(); it++) {
    if (it->first == "XRangeTest") setXRangeTest(mui, it->second);
    else if (it->first == "MeanWithinExpectedTest") setMeanWithinExpectedTest(mui, it->second);
  }
}
//
// -- Read Test Name and Parameters
//
void SiStripQualityTester::readFromFile(string fname) {
  static const int BUF_SIZE = 256;
  ifstream fin(fname.c_str(), ios::in);
  if (!fin) {
    edm::LogError("SiStripQualityTester::readFromFile") << "Input File: " 
	  << fname << " could not be opened!" << "\n";
    return;
  }
  theQTestMap.clear();
  char buf[BUF_SIZE];
  while (fin.getline(buf, BUF_SIZE, '\n')) {
    string line(buf);
    if (line.empty()) continue;
    vector<string> tokens;
    split(line, tokens);
    if (tokens.size() < 1) continue;
    cout << " The test Name and Parameters : ";
    for  (unsigned int i = 0; i < tokens.size(); i++){
      cout << tokens[i] << "  ";
    }
    cout << endl;    
    vector<string> params;
    for (unsigned int i = 1; i < tokens.size(); i++){
      params.push_back(tokens[i]);
    }
    theQTestMap.insert(pair<string, vector<string> >(tokens[0], params));
  }
}
// Attaches Quality Tests to ME's
void SiStripQualityTester::attachTests(MonitorUserInterface * mui,string path){
  for (SiStripQualityTester::TestMapType::iterator it = theQTestMap.begin();
       it != theQTestMap.end(); it++) {
    if (it->first == "XRangeTest") {
      mui->useQTest(path, it->second[0]);
    } else if (it->first == "MeanWithinExpectedTest") {
      mui->useQTest(path, it->second[0]);
    }
  }
}
// Check Status of Quality Tests
void SiStripQualityTester::checkTestResults(MonitorUserInterface * mui) {
  string currDir = mui->pwd();
  //  cout << " current Dir " << currDir << endl;
  // browse through monitorable; check if MEs exist
  if (currDir.find("detector") != string::npos)  {
    std::vector<string> contents = mui->getMEs();    
    for (std::vector<string>::const_iterator it = contents.begin();
	 it != contents.end(); it++) {
      if ((*it).find("DigisPerDetector") == 0) {
	string fullpathname = currDir + "/" + (*it); 
        MonitorElement * me = mui->get(fullpathname);
        if (me) {
	  cout << fullpathname  << endl;
	  // get all warnings associated with me
	  vector<QReport*> warnings = me->getQWarnings();
	  for(vector<QReport *>::const_iterator it = warnings.begin();
	      it != warnings.end(); ++it) {
	    edm::LogWarning("SiStripQualityTester::checkTestResults") << 
	      " *** Warning for " << me->getName() << 
	      "," << (*it)->getMessage() << "\n";
	    
	    cout <<  " *** Warning for " << me->getName() << "," 
		 << (*it)->getMessage() << " " << me->getMean() 
		 << " " << me->getRMS() << me->hasWarning() 
		 << endl;
	  }
	  // get all errors associated with me
	  vector<QReport *> errors = me->getQErrors();
	  for(vector<QReport *>::const_iterator it = errors.begin();
	      it != errors.end(); ++it) {
	    edm::LogError("SiStripQualityTester::checkTestResults") << 
	      " *** Error for " << me->getName() << 
              "," << (*it)->getMessage() << "\n";
	    
	    cout  <<   " *** Error for " << me->getName() << ","
		  << (*it)->getMessage() << " " << me->getMean() 
                  << " " << me->getRMS() 
		  << endl;
	  }
	}
      }
    }
  } else {
    std::vector<string> subdirs = mui->getSubdirs();
    for (std::vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      checkTestResults(mui);
      mui->goUp();
    }
  } 
}
//
// -- Set up ContentsXRange test with it's parameters
//
void SiStripQualityTester::setXRangeTest( MonitorUserInterface * mui, vector<string>& params){
  QCriterion * qc = mui->createQTest(ContentsXRangeROOT::getAlgoName(),params[0]);

  if (params.size() < 5) return;
  // Contents within [Xmin, Xmax]
  MEContentsXRangeROOT * me_qc = (MEContentsXRangeROOT *) qc;
  //set probability limit for test warning 
  me_qc->setWarningProb(atof(params[1].c_str()));
  //set probability limit for test error 
  me_qc->setErrorProb(atof(params[2].c_str()));
  // set allowed range in X-axis (default values: histogram's X-range)
  me_qc->setAllowedXRange(atof(params[3].c_str()), atof(params[4].c_str()));
}
//
// -- Set up MeanWithinExpected test with it's parameters
//
void SiStripQualityTester::setMeanWithinExpectedTest(MonitorUserInterface * mui, vector<string>& params){
  QCriterion * qc = mui->createQTest("MeanWithinExpected",params[0]);

  if (params.size() < 6 ) return;
  // Contents within [Xmin, Xmax]
  MEMeanWithinExpectedROOT * me_qc = (MEMeanWithinExpectedROOT *) qc;
  //set probability limit for test warning
  me_qc->setWarningProb(atof(params[1].c_str()));
  //set probability limit for test error
  me_qc->setErrorProb(atof(params[2].c_str()));
  // set Expected Mean
  me_qc->setExpectedMean(atof(params[3].c_str()));
  // set Test Type
  if (params[4] == "useRMS") me_qc->useRMS();
  else if (params[4] == "useSigma") me_qc->useSigma(atof(params[5].c_str()));
}
//
// -- Split a given string into a number of strings using given
//    delimiters and fill a vector with splitted strings
//
void SiStripQualityTester::split(const string& str, vector<string>& tokens, const string& delimiters) {
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // Find first "non-delimiter".
  string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos)  {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));

    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);

    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

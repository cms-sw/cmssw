#ifndef SiStripQualityTester_H
#define SiStripQualityTester_H

/** \class SiStripQualityTester
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2006/03/01 15:48:59 $
 *  $Revision: 1.0 $
 *  \author Suchandra Dutta
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>

using namespace std;

class SiStripQualityTester
{
 public:
  
  typedef  map<string, vector<string> > TestMapType;


  // Constructor
  SiStripQualityTester();
  
  // Destructor
  ~SiStripQualityTester();

  // Set up Quality Tests
  void setupQTests(MonitorUserInterface * mui) ;

  // Read up Quality Test Parameters from text file
  void readFromFile(string fname) ;
 
  // Attaches Quality Tests to ME's
  void attachTests(MonitorUserInterface * mui, string path) ;
  
  // Configures Test of type ContentsXRangeROOT 
  void setXRangeTest(MonitorUserInterface * mui, vector<string>& params) ;

  // Configures Test of type MeanWithinExpectedROOT
  void setMeanWithinExpectedTest(MonitorUserInterface * mui, vector<string>& params) ;

  /// Check Status of Quality Tests
  void checkTestResults(MonitorUserInterface * mui);
  

 private:
  void split(const string& str,vector<string>& tokens,const string& delimiters=" "); 
  TestMapType theQTestMap;
};

#endif

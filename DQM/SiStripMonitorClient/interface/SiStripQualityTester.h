#ifndef SiStripQualityTester_H
#define SiStripQualityTester_H

/** \class SiStripQualityTester
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2006/04/19 17:06:55 $
 *  $Revision: 1.2 $
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
  
  typedef  map<string, map<string, string> > QTestMapType;
  typedef  map<string, vector<string> > MEAssotiateMapType;


  // Constructor
  SiStripQualityTester();
  
  // Destructor
  ~SiStripQualityTester();

  // Set up Quality Tests
  void setupQTests(MonitorUserInterface * mui) ;

  // Read up Quality Test Parameters from text file
  void readQualityTests(string fname) ;
 
  // Attaches Quality Tests to ME's
  void attachTests(MonitorUserInterface * mui);
  
  // Configures Test of type ContentsXRangeROOT 
  void setXRangeTest(MonitorUserInterface * mui, string name,
        map<string, string>& params);

  // Configures Test of type MeanWithinExpectedROOT
  void setMeanWithinExpectedTest(MonitorUserInterface * mui, string name,
            map<string, string>& params);  

  int getMEsUnderTest(std::vector<std::string> & me_names);
 private:
  QTestMapType theQTestMap;
  MEAssotiateMapType theMeAssociateMap;

};

#endif

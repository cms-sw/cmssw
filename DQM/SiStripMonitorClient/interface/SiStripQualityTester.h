#ifndef SiStripQualityTester_H
#define SiStripQualityTester_H

/** \class SiStripQualityTester
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2006/05/03 09:15:01 $
 *  $Revision: 1.3 $
 *  \author Suchandra Dutta
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/ClientConfig/interface/QTestConfigurationParser.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>


class SiStripQualityTester
{
 public:
  
  typedef  std::map<std::string, std::map<std::string, std::string> > QTestMapType;
  typedef  std::map<std::string, std::vector<std::string> > MEAssotiateMapType;


  // Constructor
  SiStripQualityTester();
  
  // Destructor
  ~SiStripQualityTester();

  // Set up Quality Tests
  void setupQTests(MonitorUserInterface * mui) ;

  // Read up Quality Test Parameters from text file
  void readQualityTests(std::string fname) ;
 
  // Attaches Quality Tests to ME's
  void attachTests(MonitorUserInterface * mui);
  
  // Configures Test of type ContentsXRangeROOT 
  void setXRangeTest(MonitorUserInterface * mui, std::string name,
        std::map<std::string, std::string>& params);

  // Configures Test of type MeanWithinExpectedROOT
  void setMeanWithinExpectedTest(MonitorUserInterface * mui, std::string name,
           std::map<std::string, std::string>& params);  

  int getMEsUnderTest(std::vector<std::string> & me_names);
 private:
  QTestMapType theQTestMap;
  MEAssotiateMapType theMeAssociateMap;
};

#endif

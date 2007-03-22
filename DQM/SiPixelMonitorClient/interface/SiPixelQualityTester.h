#ifndef SiPixelQualityTester_H
#define SiPixelQualityTester_H

/** \class SiPixelQualityTester
 * *
 *  Class that handles the SiPixel Quality Tests
 * 
 *  $Date: 2006/10/19 14:11:47 $
 *  $Revision: 1.1 $
 *  \author Petra Merkel
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/ClientConfig/interface/QTestConfigurationParser.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>


class SiPixelQualityTester
{
 public:
  
  typedef  std::map<std::string, std::map<std::string, std::string> > QTestMapType;
  typedef  std::map<std::string, std::vector<std::string> > MEAssotiateMapType;


  // Constructor
  SiPixelQualityTester();
  
  // Destructor
  ~SiPixelQualityTester();

  // Set up Quality Tests
  void setupQTests(MonitorUserInterface * mui) ;

  // Read up Quality Test Parameters from text file
  void readQualityTests(std::string fname) ;
 
  // Attaches Quality Tests to ME's
  void attachTests(MonitorUserInterface * mui);
  
  // Configures Test of type ContentsXRangeROOT 
  void setXRangeTest(MonitorUserInterface * mui, std::string name,
        std::map<std::string, std::string>& params);

  // Configures Test of type ContentsYRangeROOT 
  void setYRangeTest(MonitorUserInterface * mui, std::string name,
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

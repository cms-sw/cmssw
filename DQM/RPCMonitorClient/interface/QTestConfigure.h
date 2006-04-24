#ifndef QTestConfigure_H
#define QTestConfigure_H

/** \class QTestConfigure
 * *
 *  Class that creates and defined quality tests based on
 *  the xml configuration file parsed by QTestConfigurationParser.
 *
 * 
 *  $Date: 2006/04/05 15:44:35 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include<map>
#include<vector>
#include<iostream>
#include <fstream>

class QTestConfigure{

 public:
 
  ///Constructor
  QTestConfigure(){}
  ///Destructor
  ~QTestConfigure(){}
  ///Creates and defines quality tests
  bool enableTests(std::map<std::string, std::map<std::string, std::string> > tests, MonitorUserInterface * mui); 
  ///Returns the vector containing the names of the quality tests that have been created
  std::vector<std::string> testsReady(){return testsConfigured;}
 
 private:
  ///Creates ContentsXRangeROOT test
  void EnableXRangeTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui); 
  ///Creates ContentsYRangeROOT test
  void EnableYRangeTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui); 
   ///Creates DeadChannelROOT test
  void EnableDeadChannelTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui); 
   ///Creates NoisyChannelROOT test
  void EnableNoisyChannelTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui);
    ///Creates MeanWithinExpectedROOT test
  void EnableMeanWithinExpectedTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui);

  std::vector<std::string> testsConfigured;
 
  std::ofstream logFile;  

};


#endif


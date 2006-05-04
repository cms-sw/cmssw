#ifndef QTestEnabler_H
#define QTestEnabler_H

/** \class QTestEnabler
 * *
 *  Class that creates and defined quality tests based on
 *  the xml configuration file parsed by QTestConfigurationParser.
 *
 * 
 *  $Date: 2006/04/05 08:04:04 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include<map>
#include<vector>
#include<iostream>

class QTestEnabler{

 public:
 
  ///Constructor
  QTestEnabler(){}
  ///Destructor
  ~QTestEnabler(){}
  ///Creates and defines quality tests
  bool enableTests(std::map<std::string, std::map<std::string, std::string> > tests, MonitorUserInterface * mui); 
  ///Attaches the Tests to the Monitoring element
  void startTests(std::map<std::string, std::vector<std::string> > mapMeToTests, MonitorUserInterface * mui);
  ///Returns the vector containing the names of the quality tests that have been created
  std::vector<std::string> testsReady(){return testsEnabled;}
 
 private:
  ///Creates ContentsXRangeROOT test
  void EnableXRangeTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui); 
  ///Creates ContentsYRangeROOT test
  void EnableYRangeTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui); 
   ///Creates DeadChannelROOT test
  void EnableDeadChannelTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui); 
   ///Creates NoisyChannelROOTtest
  void EnableNoisyChannelTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui);
 
  std::vector<std::string> testsEnabled;
 


};


#endif


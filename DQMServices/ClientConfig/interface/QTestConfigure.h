#ifndef QTestConfigure_H
#define QTestConfigure_H

/** \class QTestConfigure
 * *
 *  Class that creates and defined quality tests based on
 *  the xml configuration file parsed by QTestConfigurationParser.
 *
 * 
 *  $Date: 2007/05/14 13:47:25 $
 *  $Revision: 1.3 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include<map>
#include<vector>
#include<iostream>


class QTestConfigure{

 public:
 
  ///Constructor
  QTestConfigure(){}
  ///Destructor
  ~QTestConfigure(){}
  ///Creates and defines quality tests
  bool enableTests(std::map<std::string, std::map<std::string, std::string> > tests, MonitorUserInterface * mui); 
  ///Desables the Quality Tests in the string list
  void desableTests(std::vector<std::string> testsOFFList, MonitorUserInterface * mui);
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
    ///Creates MostProbableLandauROOT test
  void EnableMostProbableLandauTest( const std::string &roTEST_NAME,
                                     std::map<std::string, std::string> 
                                       &roMParams,
                                     MonitorUserInterface *poMui);

 private:
  std::vector<std::string> testsConfigured;
 

};


#endif


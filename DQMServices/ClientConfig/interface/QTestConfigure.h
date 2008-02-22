#ifndef QTestConfigure_H
#define QTestConfigure_H

/** \class QTestConfigure
 * *
 *  Class that creates and defined quality tests based on
 *  the xml configuration file parsed by QTestConfigurationParser.
 *
 * 
 *  $Date: 2008/02/21 03:26:44 $
 *  $Revision: 1.7 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/Core/interface/DQMStore.h"

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
  bool enableTests(std::map<std::string, std::map<std::string, std::string> > tests, DQMStore * bei); 
  ///Disables the Quality Tests in the string list
  void disableTests(std::vector<std::string> testsOFFList, DQMStore * bei);
  ///Returns the vector containing the names of the quality tests that have been created
  std::vector<std::string> testsReady(){return testsConfigured;}
 
 private:

  ///Creates ContentsXRangeROOT test
  void EnableXRangeTest(std::string testName, 
                        std::map<std::string, std::string>params,DQMStore * bei); 
  ///Creates ContentsYRangeROOT test
  void EnableYRangeTest(std::string testName, 
                        std::map<std::string, std::string>params,DQMStore * bei); 
   ///Creates DeadChannelROOT test
  void EnableDeadChannelTest(std::string testName, 
                             std::map<std::string,std::string> params,DQMStore * bei); 
   ///Creates NoisyChannelROOT test
  void EnableNoisyChannelTest(std::string testName, 
                              std::map<std::string,std::string> params,DQMStore * bei);
    ///Creates MeanWithinExpectedROOT test
  void EnableMeanWithinExpectedTest(std::string testName, 
                                    std::map<std::string,std::string> params,DQMStore * bei);
    ///Creates MostProbableLandauROOT test
  void EnableMostProbableLandauTest( const std::string &roTEST_NAME,
                                     std::map<std::string, std::string> &roMParams, DQMStore *bei);

    /// Creates ContentsTH2FWithinRangeROOT test
  void EnableTH2FContentsInRangeTest(std::string testName,
                                     std::map<std::string,std::string> params,DQMStore * bei);

    /// Creates ContentsProfWithinRangeROOT test
  void EnableProfContentsInRangeTest(std::string testName,
                                     std::map<std::string,std::string> params,DQMStore * bei);

    /// Creates ContentsProf2DWithinRangeROOT test
  void EnableProf2DContentsInRangeTest(std::string testName,
                                       std::map<std::string,std::string> params,DQMStore * bei);


 private:
  std::vector<std::string> testsConfigured;
 

};


#endif


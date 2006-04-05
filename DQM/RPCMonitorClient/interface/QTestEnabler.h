#ifndef QTestEnabler_H
#define QTestEnabler_H

/** \class QTestEnabler
 * *
 *  Class that intiates the DQM Quality Tests
 * 
 *  $Date: 2006/03/13 15:50:02 $
 *  $Revision: 1.3 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include<map>
#include<vector>
#include<iostream>

class QTestEnabler{

 public:
 
  QTestEnabler(){}
  ~QTestEnabler(){}
  
  void enableTests(std::map<std::string, std::map<std::string, std::string> > tests,MonitorUserInterface * mui); 
  void EnableXRangeTest(std::string testName, std::map<std::string, std::string> params,MonitorUserInterface * mui); 
  
  std::vector<std::string> testsReady(){return testsEnabled;}
 
 private:
  std::vector<std::string> testsEnabled;
 


};


#endif


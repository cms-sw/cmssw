#ifndef QTestEnabler_H
#define QTestEnabler_H

/** \class QTestEnabler
 * *
 *  Class thatattaches the quality tests to the monitoring Elements.
 *  It also subscribes to the ME's
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

class QTestEnabler{

 public:
 
  ///Constructor
  QTestEnabler(){logFile.open("QTEnabler.log");}
  ///Destructor
  ~QTestEnabler(){}
  ///Attaches the Tests to the Monitoring element
  void startTests(std::map<std::string, std::vector<std::string> > mapMeToTests, MonitorUserInterface * mui);
 
 private:
 
  std::ofstream logFile;  

};


#endif


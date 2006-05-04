#ifndef QTestEnabler_H
#define QTestEnabler_H

/** \class QTestEnabler
 * *
 *  Class that attaches the quality tests to the monitoring Elements.
 *  It also subscribes to the ME's that have Quality Tests attached
 *
 * 
 *  $Date: 2006/04/24 09:54:22 $
 *  $Revision: 1.3 $
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
  ///Attaches the Tests to the Monitoring element
  void startTests(std::map<std::string, std::vector<std::string> > mapMeToTests, MonitorUserInterface * mui);
 
 private:
 
};


#endif


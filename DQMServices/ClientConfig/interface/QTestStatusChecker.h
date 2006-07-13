#ifndef QTestStatusChecker_H
#define QTestStatusChecker_H

/** \class  QTestStatusChecker
 * *
 *  Class that checks the staus of Quality tests (takes a pointer to the
 *  MonitorUserInterface) and fills string maps containing the alarms
 *
 * 
 *  $Date: 2006/05/09 21:28:24 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include<map>
#include<string>
#include<vector>

class QTestStatusChecker{
 public:
	///Creator
	QTestStatusChecker();
	///Destructor
	~QTestStatusChecker();
	/// Check global status of Quality tests, returns a pair of string: message and color relative to global status 
	std::pair<std::string,std::string> checkGlobalStatus(MonitorUserInterface * mui); 		 
	/// Check status of quality tests for individual ME's
	std::map< std::string, std::vector<std::string> > checkDetailedStatus(MonitorUserInterface * mui);
	
	std::vector<std::string> fullPathNames(MonitorUserInterface * mui);
	void processAlarms(std::vector<std::string> allPathNames, MonitorUserInterface * mui);
 
 
 private:
 
  std::map< std::string, std::vector<std::string> > detailedWarnings;
  

};

#endif

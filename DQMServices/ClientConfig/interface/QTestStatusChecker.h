#ifndef QTestStatusChecker_H
#define QTestStatusChecker_H

/** \class  QTestStatusChecker
 * *
 *  Class that checks the staus of Quality tests (takes a pointer to the
 *  MonitorUserInterface) and fills string maps containing the alarms
 *
 * 
 *  $Date: 2008/01/11 15:47:41 $
 *  $Revision: 1.4 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
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
	std::pair<std::string,std::string> checkGlobalStatus(DaqMonitorBEInterface * bei); 		 
	/// Check status of quality tests for individual ME's
	std::map< std::string, std::vector<std::string> > checkDetailedStatus(DaqMonitorBEInterface * bei);
	
	std::vector<std::string> fullPathNames(DaqMonitorBEInterface * bei);
	void processAlarms(std::vector<std::string> allPathNames, DaqMonitorBEInterface * bei);
 
 
 private:
 
  std::map< std::string, std::vector<std::string> > detailedWarnings;
  

};

#endif

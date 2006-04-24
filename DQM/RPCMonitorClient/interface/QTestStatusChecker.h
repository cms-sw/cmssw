#ifndef QTestStatusChecker_H
#define QTestStatusChecker_H

/** \class  QTestStatusChecker
 * *
 *  Class that check the staus of Quality tests (needs a pointer to the
 *  MonitorUserInterface)
 *
 * 
 *  $Date: 2006/04/05 15:44:35 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include<map>
#include<string>
#include<vector>
#include <fstream>

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
 private:

  /// Searches ME's with tests running in all the directories
  void searchDirectories(MonitorUserInterface * mui);
  /// Check status of quality tests for individual ME's
  /// When MonitorElement.hasQualityTest() is available replace with
  /// void ProcessAlarms(MonitorElement &)
  void processAlarms(std::vector<std::string> meNames, std::string dirName, MonitorUserInterface * mui);
 
 
  std::map< std::string, std::vector<std::string> > detailedWarnings;
  
  std::ofstream logFile;  

};

#endif

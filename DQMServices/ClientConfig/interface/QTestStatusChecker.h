#ifndef QTestStatusChecker_H
#define QTestStatusChecker_H

/** \class  QTestStatusChecker
 * *
 *  Class that checks the staus of Quality tests (takes a pointer to the
 *  DQMStore) and fills string maps containing the alarms
 *
 * 
 *  $Date: 2008/02/22 23:52:23 $
 *  $Revision: 1.6 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/Core/interface/DQMStore.h"
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
	std::pair<std::string,std::string> checkGlobalStatus(DQMStore * bei); 		 
	/// Check status of quality tests for individual ME's
	std::map< std::string, std::vector<std::string> > checkDetailedStatus(DQMStore * bei);
	
	std::vector<std::string> fullPathNames(DQMStore * bei);
	void processAlarms(std::vector<std::string> allPathNames, DQMStore * bei);
 
 
 private:
 
  std::map< std::string, std::vector<std::string> > detailedWarnings;
  

};

#endif

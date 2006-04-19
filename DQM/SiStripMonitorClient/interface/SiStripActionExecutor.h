#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <map>
#include <vector>
#include <string>

class SiStripActionExecutor {

 public:

  typedef std::map< int, std::vector<std::string> > DetMapType;

  SiStripActionExecutor();
 ~SiStripActionExecutor();

 void createTkMap(MonitorUserInterface* mui,std::string me_name);
 void fillSummary(MonitorUserInterface* mui,std::string dir_name,std::string me_name);
 void checkTestResults(MonitorUserInterface * mui);


 private:
  MonitorElement* getSummaryME(MonitorUserInterface* mui, std::string me_name);
  void getValuesForTkMap(MonitorUserInterface* mui,
        std::string me_name, SiStripActionExecutor::DetMapType& values);
};
#endif

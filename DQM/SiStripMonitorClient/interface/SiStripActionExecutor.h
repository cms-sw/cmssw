#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include <map>
#include <vector>
#include <string>

using namespace std;
class SiStripActionExecutor {

 public:
  typedef map<int,vector <pair <int,float> > > DetMapType;

  SiStripActionExecutor();
 ~SiStripActionExecutor();

 void createSummary(MonitorUserInterface* mui);

 void setupQTests(MonitorUserInterface * mui);
 void checkQTestResults(MonitorUserInterface * mui);
 void createCollation(MonitorUserInterface * mui);
 void createTkMap(MonitorUserInterface* mui);
 bool readConfiguration(int& tkmap_freq, int& sum_freq);

 private:
  MonitorElement* getSummaryME(MonitorUserInterface* mui, string me_name);
  void getValuesForTkMap(MonitorUserInterface* mui,
        vector<string> me_names, SiStripActionExecutor::DetMapType& values);
  void fillSummary(MonitorUserInterface* mui, string dir_name,vector<string>& me_names);
  void drawMEs(int idet, vector<MonitorElement*>& mon_elements, vector<pair <int, float> > & values);

  SiStripConfigParser* configParser_;

};
#endif

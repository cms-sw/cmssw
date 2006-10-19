#ifndef _SiPixelActionExecutor_h_
#define _SiPixelActionExecutor_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DQMServices/Examples/interface/SiStripConfigParser.h"
//#include "DQMServices/Examples/interface/SiStripConfigWriter.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiPixelActionExecutor {

 public:
  typedef std::map<int,std::vector <std::pair <int,float> > > DetMapType;

  SiPixelActionExecutor();
 ~SiPixelActionExecutor();

 void createSummary(MonitorUserInterface* mui);

// void setupQTests(MonitorUserInterface * mui);
// void checkQTestResults(MonitorUserInterface * mui);
 void createCollation(MonitorUserInterface * mui);
 void createTkMap(MonitorUserInterface* mui);
 bool readConfiguration(int& tkmap_freq, int& sum_freq);
 void readConfiguration();
 void createLayout(MonitorUserInterface * mui);
 void fillLayout(MonitorUserInterface * mui);

 private:
  MonitorElement* getSummaryME(MonitorUserInterface* mui, std::string me_name);
  void getValuesForTkMap(MonitorUserInterface* mui,
     std::vector<std::string> me_names, SiPixelActionExecutor::DetMapType& values);
     void fillSummary(MonitorUserInterface* mui, std::string dir_name,
     std::vector<std::string>& me_names);
  void drawMEs(int idet, std::vector<MonitorElement*>& mon_elements, 
                    std::vector<std::pair <int, float> > & values);

//  SiPixelConfigParser* configParser_;
//  SiPixelConfigWriter* configWriter_;
};
#endif

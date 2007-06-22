#ifndef _SiStripSummaryCreator_h_
#define _SiStripSummaryCreator_h_

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiStripSummaryCreator {

 public:

  SiStripSummaryCreator();
  virtual ~SiStripSummaryCreator();

  void createSummary(MonitorUserInterface* mui);

  void createLayout(MonitorUserInterface * mui);
  void fillLayout(MonitorUserInterface * mui);
  void setSummaryMENames( std::map<std::string, std::string>& me_names);

 private:
 MonitorElement* getSummaryME(MonitorUserInterface* mui,
                              std::string& name, std::string htype);


  void fillGrandSummaryHistos(MonitorUserInterface* mui);
  void fillSummaryHistos(MonitorUserInterface* mui);
  void fillHistos(int ival, int istep, std::string htype, 
		  MonitorElement* me_src, MonitorElement* me);
  std::map<std::string, std::string> summaryMEMap;

  SiStripConfigWriter* configWriter_;
};
#endif

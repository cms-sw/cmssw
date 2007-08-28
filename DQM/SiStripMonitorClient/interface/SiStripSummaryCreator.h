#ifndef _SiStripSummaryCreator_h_
#define _SiStripSummaryCreator_h_

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>


class SiStripConfigWriter;

class SiStripSummaryCreator {

 public:

  SiStripSummaryCreator();
  virtual ~SiStripSummaryCreator();
  bool readConfiguration();

  void createSummary(DaqMonitorBEInterface* bei);

  void createLayout(DaqMonitorBEInterface * bei);
  void fillLayout(DaqMonitorBEInterface * bei);
  void setSummaryMENames( std::map<std::string, std::string>& me_names);
  int getFrequency() { return summaryFrequency_;}

 private:
 MonitorElement* getSummaryME(DaqMonitorBEInterface* bei,
                              std::string& name, std::string htype);


  void fillGrandSummaryHistos(DaqMonitorBEInterface* bei);
  void fillSummaryHistos(DaqMonitorBEInterface* bei);
  void fillHistos(int ival, int istep, std::string htype, 
		  MonitorElement* me_src, MonitorElement* me);


  std::map<std::string, std::string> summaryMEMap;

  
  SiStripConfigWriter* configWriter_;
  int summaryFrequency_;


};
#endif

#ifndef _SiStripSummaryCreator_h_
#define _SiStripSummaryCreator_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>


class SiStripConfigWriter;
class DQMStore;

class SiStripSummaryCreator {

 public:

  SiStripSummaryCreator();
  virtual ~SiStripSummaryCreator();
  bool readConfiguration(std::string & file_path);

  void createSummary(DQMStore* dqm_store);

  void fillLayout(DQMStore * dqm_store);
  void setSummaryMENames( std::map<std::string, std::string>& me_names);
  int getFrequency() { return summaryFrequency_;}

 private:
 MonitorElement* getSummaryME(DQMStore* dqm_store,
                              std::string& name, std::string htype);


  void fillGrandSummaryHistos(DQMStore* dqm_store);
  void fillSummaryHistos(DQMStore* dqm_store);
  void fillHistos(int ival, int istep, std::string htype, 
		  MonitorElement* me_src, MonitorElement* me);


  std::map<std::string, std::string> summaryMEMap;
 
  int summaryFrequency_;


};
#endif

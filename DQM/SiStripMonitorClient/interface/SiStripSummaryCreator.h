#ifndef _SiStripSummaryCreator_h_
#define _SiStripSummaryCreator_h_

#include "DQMServices/Core/interface/DQMStore.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>

class SiStripConfigWriter;

class SiStripSummaryCreator {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripSummaryCreator();
  virtual ~SiStripSummaryCreator();
  bool readConfiguration(std::string const& file_path);

  void createSummary(DQMStore& dqm_store);

  void fillLayout(DQMStore& dqm_store);
  void setSummaryMENames(std::map<std::string, std::string>& me_names);
  int getFrequency() const { return summaryFrequency_; }

private:
  MonitorElement* getSummaryME(DQMStore& dqm_store, std::string& name, std::string htype);

  void fillGrandSummaryHistos(DQMStore& dqm_store);
  void fillSummaryHistos(DQMStore& dqm_store);
  void fillHistos(int ival, int istep, std::string htype, MonitorElement* me_src, MonitorElement* me);

  std::map<std::string, std::string> summaryMEs_;
  int summaryFrequency_{-1};
};
#endif
